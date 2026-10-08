"""Leaderboard predictions for flan-t5-xl (LoRA) standalone -- no blending
with exp09. First-ever real test of a non-DeBERTa architecture in isolation;
every real submission so far has been exp09 or an exp09-derived blend.

True-OOF signal: unweighted WF1 0.8038 (+0.0024 vs exp09's 0.8014), but the
importance-weighted retrospective check (population-shift correction) flips
this to -0.0069 vs exp09 -- flagged as likely to regress, same direction as
every other candidate tried this session. Built anyway per explicit request,
to get the real data point.

Usage:
    python build_flant5xl_solo_submission.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, ID2RISK, DATA_DIR, RESULTS_DIR

MODEL_NAME = "google/flan-t5-xl"
MAX_SRC_LEN = 512
N_FOLDS = 5
EXP_NAME = "exp_flant5_xl_qa_sft_lora"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)

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
            batch_prompts = prompts[i:i + 16]
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
            all_scores[i:i + bs, li] = seq_lp
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
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model = PeftModel.from_pretrained(base_model, fold_dir).to(device)
        model.eval()
        probs = scored_probs_one_fold(model, tokenizer, prompts, device)
        fold_probs.append(probs)
        print(f"  flan-t5-xl fold {fold} done")
        del model, base_model
        torch.cuda.empty_cache()

    lb_probs = np.mean(fold_probs, axis=0)
    np.save(os.path.join(RESULTS_DIR, "flant5xl_solo_leaderboard_probs.npy"), lb_probs)

    pred_ids = lb_probs.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    print("\nPrediction distribution (flan-t5-xl solo):")
    print(pd.Series(pred_labels).value_counts().to_string())

    exp09_logits = np.load(os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high",
                                          "sequence_data", "leaderboard_logits.npy"))
    exp09_labels = [ID2RISK[i].capitalize() for i in exp09_logits.argmax(axis=1)]
    n_changed = sum(a != b for a, b in zip(pred_labels, exp09_labels))
    print(f"\nRows where flan-t5-xl solo label differs from exp09-alone: {n_changed}/{len(lb_df)}")

    out = pd.DataFrame({"row_id": lb_df["row_id"], "risk_level": pred_labels})
    out_path = os.path.join(RESULTS_DIR, "flant5xl_solo_1a_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved 1a predictions -> {out_path}")


if __name__ == "__main__":
    main()
