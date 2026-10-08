"""Subtask 1a: true-OOF error analysis for exp09 (the confirmed-best real
production model). Loads each fold's already-trained best_fold{N}.pt
checkpoint and runs inference ONLY on that fold's own held-out validation
split (same GroupKFold(n_splits=5) call as trainer.py, deterministic given
anon_user_id groups -- no shuffling involved, so this reproduces the exact
same splits used during training). No retraining, no leakage: every
prediction comes from a model that never saw that post during training.

Produces:
  - confusion matrix + per-class precision/recall/F1
  - per-fold WF1 (sanity check against the original training log)
  - error rate by post length bucket
  - full per-post predictions CSV (for manual reading of misclassified posts)
  - the N highest-confidence WRONG predictions per (true, pred) confusion pair
    -- these are the most informative errors: not borderline/low-confidence
    mistakes, but posts the model was confidently wrong about.

Usage:
    python error_analysis_exp09.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, load_data, TextDataset, RISK_LEVELS, RISK2ID, ID2RISK, RESULTS_DIR
from transformers import AutoTokenizer

EXP_NAME = "exp09_deberta_large_lr_high"
EXP_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 32
N_FOLDS = 5

OUT_DIR = os.path.join(RESULTS_DIR, "error_analysis_exp09")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def run_fold_inference(fold, va_df, tokenizer, device):
    ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
    model = Classifier(MODEL_NAME).to(device).float()
    model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
    model.eval()

    ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                      tokenizer, MAX_LENGTH)
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_logits = []
    for batch in ld:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(ids, mask)
        all_logits.append(logits.float().cpu())
    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(all_logits, dim=-1).numpy()

    del model
    torch.cuda.empty_cache()
    return probs


def length_bucket(post):
    n = len(str(post).split())
    if n < 30:
        return "short (<30 words)"
    elif n < 80:
        return "medium (30-80 words)"
    else:
        return "long (80+ words)"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = load_data("original")  # exp09's exact data mode -- no oversampling, no synthetic rows
    df = df.reset_index(drop=True)
    print(f"Dataset: {len(df)} posts")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    all_rows = []
    fold_wf1s = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i].reset_index(drop=True)
        print(f"\nFold {fold+1}/{N_FOLDS}: {len(va_df)} val posts")

        probs = run_fold_inference(fold, va_df, tokenizer, device)
        preds = probs.argmax(axis=1)
        true = np.array([RISK2ID[r] for r in va_df["risk_norm"]])

        wf1 = f1_score(true, preds, average="weighted", zero_division=0)
        fold_wf1s.append(wf1)
        print(f"  Fold {fold+1} WF1={wf1:.4f} (training_log had: check summary.json)")

        for i in range(len(va_df)):
            row = va_df.iloc[i]
            all_rows.append({
                "fold": fold,
                "row_id": row.get("row_id", ""),
                "post": row["post"],
                "n_words": len(str(row["post"]).split()),
                "length_bucket": length_bucket(row["post"]),
                "true_label": row["risk_norm"],
                "pred_label": ID2RISK[preds[i]],
                "correct": row["risk_norm"] == ID2RISK[preds[i]],
                "confidence": float(probs[i, preds[i]]),
                "prob_indicator": float(probs[i, RISK2ID["indicator"]]),
                "prob_ideation": float(probs[i, RISK2ID["ideation"]]),
                "prob_behavior": float(probs[i, RISK2ID["behavior"]]),
                "prob_attempt": float(probs[i, RISK2ID["attempt"]]),
            })

    print(f"\nMean fold WF1: {np.mean(fold_wf1s):.4f} +/- {np.std(fold_wf1s):.4f}")
    print("(compare against results/exp09_deberta_large_lr_high/summary.json's "
          "fold_wf1 -- should match closely, confirms this reproduces the exact "
          "same true-OOF split/checkpoint pairing used during training)")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    true_all = result_df["true_label"].tolist()
    pred_all = result_df["pred_label"].tolist()
    overall_wf1 = f1_score(true_all, pred_all, labels=RISK_LEVELS, average="weighted", zero_division=0)

    print(f"\n{'='*70}\n  OVERALL TRUE-OOF WF1: {overall_wf1:.4f}\n{'='*70}")

    print("\n--- Per-class report ---")
    report = classification_report(true_all, pred_all, labels=RISK_LEVELS, zero_division=0)
    print(report)

    print("\n--- Confusion matrix (rows=true, cols=pred) ---")
    cm = confusion_matrix(true_all, pred_all, labels=RISK_LEVELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{r}" for r in RISK_LEVELS],
                          columns=[f"pred_{r}" for r in RISK_LEVELS])
    print(cm_df.to_string())
    cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

    print("\n--- Error rate by post length ---")
    for bucket in ["short (<30 words)", "medium (30-80 words)", "long (80+ words)"]:
        sub = result_df[result_df["length_bucket"] == bucket]
        if len(sub):
            acc = sub["correct"].mean()
            print(f"  {bucket:25s} n={len(sub):4d}  accuracy={acc:.4f}")

    print("\n--- Error rate by fold ---")
    for fold in range(N_FOLDS):
        sub = result_df[result_df["fold"] == fold]
        acc = sub["correct"].mean()
        print(f"  Fold {fold}  n={len(sub):4d}  accuracy={acc:.4f}")

    print("\n--- Most confident WRONG predictions per (true, pred) pair ---")
    wrong = result_df[~result_df["correct"]].copy()
    top_errors = []
    for (true_l, pred_l), grp in wrong.groupby(["true_label", "pred_label"]):
        top = grp.nlargest(3, "confidence")
        top_errors.append(top)
        print(f"\n  true={true_l} -> pred={pred_l}  (n={len(grp)} total confusions of this type)")
        for _, r in top.iterrows():
            snippet = str(r["post"])[:200].replace("\n", " ")
            print(f"    conf={r['confidence']:.3f}  \"{snippet}...\"")

    if top_errors:
        pd.concat(top_errors).to_csv(os.path.join(OUT_DIR, "top_confident_errors.csv"), index=False)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "overall_true_oof_wf1": float(overall_wf1),
            "fold_wf1": [float(w) for w in fold_wf1s],
            "n_posts": len(result_df),
            "n_errors": int((~result_df["correct"]).sum()),
            "error_rate_by_length": {
                b: float(result_df[result_df["length_bucket"] == b]["correct"].mean())
                for b in result_df["length_bucket"].unique()
            },
        }, f, indent=2)

    print(f"\nSaved: {OUT_DIR}/oof_predictions.csv, confusion_matrix.csv, "
          f"top_confident_errors.csv, summary.json")


if __name__ == "__main__":
    main()
