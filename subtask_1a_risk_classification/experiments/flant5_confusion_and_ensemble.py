"""Two things in one pass, both building on the already-verified scored-decoding
setup (rescore_flant5_large_constrained.py -- split reproduction confirmed
0.0000 diff against the original training log, safe because flan-t5-large was
trained in this same session/environment):

1. Flan-T5-large's OWN true-OOF confusion matrix. Error analyses so far
   (exp09's) don't necessarily transfer to a different architecture -- this
   checks whether flan-t5 actually fails on the same cases exp09 does, rather
   than assuming it.

2. Ensemble flan-t5-large (scored-decoding probabilities) with exp09
   (already-verified true-OOF probabilities from train_oof_logits.npy).
   Motivation: every attempt to beat exp09 with flan-t5 ALONE has failed
   (three clean variants, consistent ~-0.023 to -0.025 gap). But flan-t5 is
   architecturally very different from exp09's DeBERTa backbone -- different
   pretraining, different architecture family (encoder-decoder vs
   encoder-only) -- unlike every other Sub1a ensemble member tried so far,
   which are all DeBERTa variants (FGM/AWP/SupCon/CORAL/etc.) with likely
   correlated errors. This tests whether flan-t5's errors are different
   enough from exp09's to add ensemble value despite being individually
   weaker.

Row alignment note: exp09's OOF (train_oof_logits.npy) and flan-t5's OOF
(recomputed here) come from splits derived at DIFFERENT times (exp09: Aug 6,
old environment; flan-t5: today). Their fold ASSIGNMENTS might not match
each other (that's fine, doesn't matter) -- what matters is that each is
independently a genuine true-OOF prediction for its own row (verified
separately for each), so combining them per-row is valid regardless of
whether the two splits coincide. Both scripts read train.xlsx directly,
unfiltered, in the same deterministic row order, so alignment by row index
is safe.

Usage:
    python flant5_confusion_and_ensemble.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, RISK2ID, ID2RISK, DATA_DIR, RESULTS_DIR

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"
MAX_SRC_LEN = 512
N_FOLDS = 5
CKPT_DIR = os.path.join(RESULTS_DIR, "exp_flant5_large_qa_sft")  # bare-prompt, best solo variant

PROMPT_TEMPLATE = (
    "Classify the suicide risk level expressed in the following social media "
    "post into exactly one of these four levels, ordered from lowest to "
    "highest risk: indicator, ideation, behavior, attempt.\n\n"
    "Post: {post}\n\n"
    "Risk level:"
)


@torch.no_grad()
def scored_probs(model, tokenizer, prompts, device):
    """Returns (N, 4) softmax probabilities over the 4 candidate labels,
    via teacher-forced log-prob scoring (no generation)."""
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

    # Softmax over the 4 candidate scores -> proper probability distribution
    exp_scores = np.exp(all_scores - all_scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    flant5_probs = np.zeros((len(df), 4))

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i].reset_index(drop=True)
        prompts = [PROMPT_TEMPLATE.format(post=p) for p in va_df["post"].astype(str)]

        ckpt_path = os.path.join(CKPT_DIR, f"best_fold{fold}.pt")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
        model.eval()

        probs = scored_probs(model, tokenizer, prompts, device)
        flant5_probs[va_i] = probs
        print(f"  Fold {fold}: {len(va_df)} rows scored")

        del model
        torch.cuda.empty_cache()

    flant5_preds = flant5_probs.argmax(axis=1)
    flant5_wf1 = f1_score(true_labels, flant5_preds, average="weighted", zero_division=0)
    print(f"\nFlan-T5-large solo true-OOF WF1 (sanity check, should be ~0.7784): {flant5_wf1:.4f}")

    # ── Part 1: flan-t5's own confusion matrix ──────────────────────────────
    print(f"\n{'='*70}\n  FLAN-T5-LARGE CONFUSION MATRIX (true OOF)\n{'='*70}")
    true_str = [ID2RISK[t] for t in true_labels]
    pred_str = [ID2RISK[p] for p in flant5_preds]
    print(classification_report(true_str, pred_str, labels=RISK_LEVELS, zero_division=0))
    cm = confusion_matrix(true_str, pred_str, labels=RISK_LEVELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{r}" for r in RISK_LEVELS],
                          columns=[f"pred_{r}" for r in RISK_LEVELS])
    print(cm_df.to_string())

    # For direct comparison, also print exp09's confusion matrix side by side
    exp09_logits = np.load(os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high",
                                          "sequence_data", "train_oof_logits.npy"))
    exp09_probs_raw = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)
    exp09_preds = exp09_probs_raw.argmax(axis=1)
    exp09_wf1 = f1_score(true_labels, exp09_preds, average="weighted", zero_division=0)
    print(f"\nexp09 solo true-OOF WF1 (sanity check, should be ~0.8014): {exp09_wf1:.4f}")
    print(f"\n{'='*70}\n  exp09 CONFUSION MATRIX (true OOF, for comparison)\n{'='*70}")
    exp09_pred_str = [ID2RISK[p] for p in exp09_preds]
    cm2 = confusion_matrix(true_str, exp09_pred_str, labels=RISK_LEVELS)
    print(pd.DataFrame(cm2, index=[f"true_{r}" for r in RISK_LEVELS],
                        columns=[f"pred_{r}" for r in RISK_LEVELS]).to_string())

    # Agreement / disagreement analysis -- how correlated are the two models' errors?
    both_correct = (flant5_preds == true_labels) & (exp09_preds == true_labels)
    both_wrong = (flant5_preds != true_labels) & (exp09_preds != true_labels)
    only_flant5_right = (flant5_preds == true_labels) & (exp09_preds != true_labels)
    only_exp09_right = (flant5_preds != true_labels) & (exp09_preds == true_labels)
    n = len(true_labels)
    print(f"\n{'='*70}\n  ERROR CORRELATION\n{'='*70}")
    print(f"  Both correct:        {both_correct.sum():4d} ({100*both_correct.sum()/n:.1f}%)")
    print(f"  Both wrong:          {both_wrong.sum():4d} ({100*both_wrong.sum()/n:.1f}%)")
    print(f"  Only flan-t5 right:  {only_flant5_right.sum():4d} ({100*only_flant5_right.sum()/n:.1f}%)  <- exp09's errors flan-t5 could fix")
    print(f"  Only exp09 right:    {only_exp09_right.sum():4d} ({100*only_exp09_right.sum()/n:.1f}%)")
    if (exp09_preds != true_labels).sum() > 0:
        rescue_rate = only_flant5_right.sum() / (exp09_preds != true_labels).sum()
        print(f"  Of exp09's {(exp09_preds != true_labels).sum()} errors, flan-t5 gets "
              f"{only_flant5_right.sum()} right ({100*rescue_rate:.1f}% potential rescue rate)")

    # ── Part 2: ensemble ────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  ENSEMBLE: exp09 + flan-t5-large\n{'='*70}")
    for w_flant5 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w_exp09 = 1 - w_flant5
        ens_probs = w_exp09 * exp09_probs_raw + w_flant5 * flant5_probs
        ens_preds = ens_probs.argmax(axis=1)
        ens_wf1 = f1_score(true_labels, ens_preds, average="weighted", zero_division=0)
        print(f"  weight(flan-t5)={w_flant5:.1f}  weight(exp09)={w_exp09:.1f}  "
              f"ensemble WF1 = {ens_wf1:.4f}  (vs exp09 alone: {ens_wf1 - exp09_wf1:+.4f})")

    results = {
        "flant5_solo_wf1": float(flant5_wf1),
        "exp09_solo_wf1": float(exp09_wf1),
        "both_correct_pct": float(both_correct.mean()),
        "both_wrong_pct": float(both_wrong.mean()),
        "only_flant5_right_pct": float(only_flant5_right.mean()),
        "only_exp09_right_pct": float(only_exp09_right.mean()),
        "ensemble_by_weight": {},
    }
    for w_flant5 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w_exp09 = 1 - w_flant5
        ens_probs = w_exp09 * exp09_probs_raw + w_flant5 * flant5_probs
        ens_wf1 = f1_score(true_labels, ens_probs.argmax(axis=1), average="weighted", zero_division=0)
        results["ensemble_by_weight"][str(w_flant5)] = float(ens_wf1)

    np.save(os.path.join(CKPT_DIR, "train_oof_scored_probs.npy"), flant5_probs)
    with open(os.path.join(CKPT_DIR, "confusion_and_ensemble_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved probabilities and results to {CKPT_DIR}/")


if __name__ == "__main__":
    main()
