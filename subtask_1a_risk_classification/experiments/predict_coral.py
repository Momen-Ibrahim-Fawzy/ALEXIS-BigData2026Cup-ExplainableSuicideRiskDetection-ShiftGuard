"""
Evaluate the CORAL dual-head model (backlog item #2) via true OOF, and
generate leaderboard predictions if it beats the exp09 baseline.

Usage:
    python predict_coral.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import TextDataset, RISK2ID, ID2RISK, RISK_LEVELS, BASE_DIR, DATA_DIR, RESULTS_DIR
from coral_model import CoralDualHeadClassifier
from train_coral import EXP_NAME, MODEL_NAME, MAX_LENGTH, N_FOLDS, EXP_DIR

SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

# Baseline to compare against: exp09's true OOF WF1 (see tune_thresholds_oof.py's
# earlier run, and this repo's EXPERIMENT_REPORT.md)
EXP09_BASELINE_WF1 = 0.8014


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    all_preds = np.full(len(df), -1, dtype=np.int64)
    all_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold}: checkpoint not found, skipping (incomplete training).")
            continue
        print(f"  [OOF] fold {fold} <- {ckpt_path}")

        model = CoralDualHeadClassifier(MODEL_NAME).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        va_ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                             tokenizer, MAX_LENGTH)
        va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        preds = []
        with torch.no_grad():
            for batch in va_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                standard_logits, coral_logits = model(ids, mask)
                probs = model.combined_probs(standard_logits.float(), coral_logits.float())
                preds.extend(probs.argmax(-1).cpu().tolist())
        all_preds[va_i] = preds
        del model
        torch.cuda.empty_cache()

    if (all_preds == -1).any():
        print("Not all folds trained yet — cannot compute full OOF. Exiting.")
        return

    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"\nCORAL dual-head true OOF  WF1={wf1:.4f}  MF1={mf1:.4f}")
    print(f"exp09 baseline            WF1={EXP09_BASELINE_WF1:.4f}")
    print(f"Improvement:              {wf1 - EXP09_BASELINE_WF1:+.4f}")

    result = {"coral_oof_wf1": float(wf1), "coral_oof_mf1": float(mf1),
              "exp09_baseline_wf1": EXP09_BASELINE_WF1, "improvement": float(wf1 - EXP09_BASELINE_WF1)}
    with open(os.path.join(EXP_DIR, "oof_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if wf1 <= EXP09_BASELINE_WF1:
        print("\n*** CORAL dual-head did NOT beat exp09 baseline on true OOF. "
              "Do not use it for a submission. ***")
        return

    # ── Leaderboard prediction (5-fold ensemble average of combined_probs) ──
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    lb_ds = TextDataset(lb_df["post"].tolist(), [0] * len(lb_df), tokenizer, MAX_LENGTH)
    lb_ld = DataLoader(lb_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        model = CoralDualHeadClassifier(MODEL_NAME).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        probs_list = []
        with torch.no_grad():
            for batch in lb_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                standard_logits, coral_logits = model(ids, mask)
                probs = model.combined_probs(standard_logits.float(), coral_logits.float())
                probs_list.append(probs.cpu().numpy())
        fold_probs.append(np.concatenate(probs_list, axis=0))
        del model
        torch.cuda.empty_cache()

    avg_probs = np.mean(fold_probs, axis=0)
    pred_ids = avg_probs.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    submission = pd.DataFrame({"row_id": lb_df["row_id"], "risk_level": pred_labels})
    print("\nLeaderboard prediction distribution:")
    print(submission["risk_level"].value_counts().to_string())

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_exp25_deberta_coral_ordinal.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
