"""
Sub1a post-hoc per-class calibration for exp09_deberta_large_lr_high.

Reconstructs the exact GroupKFold OOF split used during training, runs each
fold's own checkpoint on its held-out validation rows (never seen in training),
collects true out-of-fold probabilities for all 1635 posts, then searches for
a per-class logit bias vector that maximizes Weighted F1 on that OOF set.

The tuned bias is then applied to leaderboard logits before argmax — same
mechanism as tuning per-factor thresholds in Sub2, adapted for single-label
4-way classification (bias-shift instead of independent thresholds, since
classes compete via argmax rather than independent sigmoid decisions).

Usage:
    python tune_thresholds_oof.py
"""

import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, ID2RISK, RISK_LEVELS, BASE_DIR, DATA_DIR, RESULTS_DIR
from experiment_config import EXPERIMENTS

EXP_NAME = "exp09_deberta_large_lr_high"
N_FOLDS = 5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EXPERIMENTS[EXP_NAME]
    exp_dir = os.path.join(RESULTS_DIR, EXP_NAME)

    import pandas as pd
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    # exp09 uses data="original" -> no oversample rows, orig_df == df
    orig_df = df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = orig_df["anon_user_id"].values

    oof_probs = np.zeros((len(orig_df), 4), dtype=np.float64)
    oof_labels = np.full(len(orig_df), -1, dtype=np.int64)
    filled = np.zeros(len(orig_df), dtype=bool)

    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        va_df = orig_df.iloc[va_i]
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        print(f"  Fold {fold}: {len(va_i)} val rows, loading {ckpt_path}")

        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        va_ds = TextDataset(va_df["post"].tolist(),
                             [RISK2ID[r] for r in va_df["risk_norm"]],
                             tokenizer, cfg["max_length"])
        va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in va_ld:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                logits = model(ids, mask)
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                probs_list.append(probs)
                labels_list.append(labels.numpy())

        probs = np.concatenate(probs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        oof_probs[va_i] = probs
        oof_labels[va_i] = labels
        filled[va_i] = True

        del model
        torch.cuda.empty_cache()

    assert filled.all(), "Not all rows received an OOF prediction"

    # Baseline: raw argmax
    base_preds = oof_probs.argmax(axis=1)
    base_wf1 = f1_score(oof_labels, base_preds, average="weighted", zero_division=0)
    base_mf1 = f1_score(oof_labels, base_preds, average="macro", zero_division=0)
    print(f"\n  Baseline (argmax)   WF1={base_wf1:.4f}  MF1={base_mf1:.4f}")

    # Search for a per-class log-bias vector that maximizes WF1.
    # Bias is shift-invariant (only relative differences matter), so fix
    # bias[0]=0 (indicator) and search the other 3 classes.
    logp = np.log(np.clip(oof_probs, 1e-9, 1.0))

    best_wf1 = base_wf1
    best_bias = np.zeros(4)

    rng = np.arange(-2.0, 2.01, 0.1)
    # Coordinate ascent: cycle through classes 1,2,3 (indicator fixed at 0),
    # repeat a few passes since classes interact via softmax competition.
    bias = np.zeros(4)
    for pass_i in range(4):
        improved = False
        for c in [1, 2, 3]:
            local_best_wf1 = best_wf1
            local_best_val = bias[c]
            for val in rng:
                trial = bias.copy()
                trial[c] = val
                preds = (logp + trial).argmax(axis=1)
                wf1 = f1_score(oof_labels, preds, average="weighted", zero_division=0)
                if wf1 > local_best_wf1:
                    local_best_wf1 = wf1
                    local_best_val = val
            if local_best_wf1 > best_wf1 + 1e-6:
                bias[c] = local_best_val
                best_wf1 = local_best_wf1
                improved = True
        print(f"  Pass {pass_i}: bias={bias.round(2).tolist()}  WF1={best_wf1:.4f}")
        if not improved:
            break

    best_bias = bias
    final_preds = (logp + best_bias).argmax(axis=1)
    final_wf1 = f1_score(oof_labels, final_preds, average="weighted", zero_division=0)
    final_mf1 = f1_score(oof_labels, final_preds, average="macro", zero_division=0)

    print(f"\n  Baseline WF1: {base_wf1:.4f}  MF1: {base_mf1:.4f}")
    print(f"  Tuned    WF1: {final_wf1:.4f}  MF1: {final_mf1:.4f}")
    print(f"  Improvement:  {final_wf1 - base_wf1:+.4f} WF1")
    print(f"  Bias vector (indicator, ideation, behavior, attempt): {best_bias.round(3).tolist()}")

    out = {
        "exp_name": EXP_NAME,
        "risk_levels": RISK_LEVELS,
        "baseline_wf1": float(base_wf1),
        "baseline_mf1": float(base_mf1),
        "tuned_wf1": float(final_wf1),
        "tuned_mf1": float(final_mf1),
        "improvement": float(final_wf1 - base_wf1),
        "bias_vector": best_bias.tolist(),
    }
    out_path = os.path.join(exp_dir, "threshold_calibration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
