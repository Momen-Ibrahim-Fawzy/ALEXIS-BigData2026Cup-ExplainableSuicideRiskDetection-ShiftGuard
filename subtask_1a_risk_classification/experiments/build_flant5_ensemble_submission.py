"""Leaderboard predictions for the exp09 + flan-t5-large (scored-decoding)
ensemble at weight(flan-t5)=0.10.

Selected because, unlike every other Sub1a ensemble candidate this project has
tried (all DeBERTa variants of exp09 -- FGM/AWP/SupCon/CORAL/etc., likely
correlated errors), flan-t5-large is a genuinely different architecture
(encoder-decoder vs encoder-only, different pretraining). True-OOF error
correlation check: of exp09's 327 true-OOF errors, flan-t5-large independently
gets 119 right (36.4% rescue rate) -- real, substantial diversity, not just a
weaker copy of the same model. True-OOF ensemble sweep (train.xlsx, all
weights 0.025-0.40) was positive everywhere in that range, strongest and most
stable at 0.075-0.125 (WF1 0.8055-0.8062, exp09 alone = 0.8014); w=0.10 chosen
as the round-number midpoint of that stable region rather than the single-best
point, to avoid overfitting the exact weight to this one offline check.

flan-t5's leaderboard probabilities are computed the same way as its true-OOF
ones (rescore_flant5_large_constrained.py / flant5_confusion_and_ensemble.py):
teacher-forced scoring of all 4 candidate labels, softmax over the 4 scores --
NOT free-form generation (already verified byte-identical to generation on
this checkpoint, so equivalent either way, but scoring is what's already
validated). 5-fold-averaged, same convention as every other production
inference in this project.

Usage:
    python build_flant5_ensemble_submission.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, ID2RISK, DATA_DIR, RESULTS_DIR

MODEL_NAME = "google/flan-t5-large"
MAX_SRC_LEN = 512
N_FOLDS = 5
CKPT_DIR = os.path.join(RESULTS_DIR, "exp_flant5_large_qa_sft")
W_FLANT5 = 0.10

PROMPT_TEMPLATE = (
    "Classify the suicide risk level expressed in the following social media "
    "post into exactly one of these four levels, ordered from lowest to "
    "highest risk: indicator, ideation, behavior, attempt.\n\n"
    "Post: {post}\n\n"
    "Risk level:"
)


@torch.no_grad()
def scored_probs_one_fold(model, tokenizer, prompts, device):
    label_ids = {label: tokenizer(label, return_tensors="pt")["input_ids"][0] for label in RISK_LEVELS}
    all_scores = np.zeros((len(prompts), len(RISK_LEVELS)))
    for li, label in enumerate(RISK_LEVELS):
        tgt_ids = label_ids[label]
        for i in range(0, len(prompts), 16):
            batch_prompts = prompts[i:i+16]
            enc = tokenizer(batch_prompts, max_length=MAX_SRC_LEN, padding=True,
                             truncation=True, return_tensors="pt").to(device)
            bs = len(batch_prompts)
            decoder_input = tgt_ids.unsqueeze(0).repeat(bs, 1).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=decoder_input)
            log_probs = torch.log_softmax(out.logits.float(), dim=-1)
            tgt = decoder_input.unsqueeze(-1)
            token_lp = torch.gather(log_probs, 2, tgt).squeeze(-1)
            seq_lp = token_lp.sum(dim=1).cpu().numpy()
            all_scores[i:i+bs, li] = seq_lp
    exp_scores = np.exp(all_scores - all_scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    print(f"Leaderboard posts: {len(lb_df)}")
    prompts = [PROMPT_TEMPLATE.format(post=p) for p in lb_df["post"].astype(str)]

    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(CKPT_DIR, f"best_fold{fold}.pt")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
        model.eval()
        probs = scored_probs_one_fold(model, tokenizer, prompts, device)
        fold_probs.append(probs)
        print(f"  flan-t5 fold {fold} done")
        del model
        torch.cuda.empty_cache()

    flant5_lb_probs = np.mean(fold_probs, axis=0)
    np.save(os.path.join(CKPT_DIR, "leaderboard_scored_probs.npy"), flant5_lb_probs)

    exp09_logits = np.load(os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high",
                                          "sequence_data", "leaderboard_logits.npy"))
    exp09_lb_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)

    ensemble_probs = (1 - W_FLANT5) * exp09_lb_probs + W_FLANT5 * flant5_lb_probs
    np.save(os.path.join(RESULTS_DIR, "flant5_ensemble_leaderboard_probs.npy"), ensemble_probs)

    pred_ids = ensemble_probs.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    print("\nPrediction distribution (ensemble):")
    print(pd.Series(pred_labels).value_counts().to_string())

    exp09_only_labels = [ID2RISK[i].capitalize() for i in exp09_lb_probs.argmax(axis=1)]
    n_changed = sum(a != b for a, b in zip(pred_labels, exp09_only_labels))
    print(f"\nRows where ensemble label differs from exp09-alone: {n_changed}/{len(lb_df)}")

    out = pd.DataFrame({"row_id": lb_df["row_id"], "risk_level": pred_labels})
    out_path = os.path.join(RESULTS_DIR, "flant5_ensemble_1a_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved 1a predictions -> {out_path}")


if __name__ == "__main__":
    main()
