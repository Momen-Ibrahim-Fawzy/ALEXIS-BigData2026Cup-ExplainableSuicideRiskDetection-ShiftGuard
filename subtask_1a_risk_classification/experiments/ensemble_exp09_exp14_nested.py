"""
Honest nested-CV check on the exp09+exp14 blend weight: ensemble_exp09_exp14.py
grid-searches the blend weight on the full 1635-row true-OOF set and scores it
on that same set — a mild form of leakage (this project has been burned before
by OOF gains that didn't generalize, e.g. the Sub2 3-way ensemble and Sub2
calibration). This script tunes the weight on 4 folds and scores only on the
held-out 5th fold, for every fold, so the reported number is honestly
out-of-sample end to end.

Usage:
    python ensemble_exp09_exp14_nested.py
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, RESULTS_DIR

EXP09_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
EXP14_DIR = os.path.join(RESULTS_DIR, "exp14_mental_roberta_large", "sequence_data")
N_FOLDS = 5


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    train_meta = pd.read_csv(os.path.join(EXP09_DIR, "train_meta.csv"))
    exp09_probs = softmax(np.load(os.path.join(EXP09_DIR, "train_oof_logits.npy")))
    exp14_probs = softmax(np.load(os.path.join(EXP14_DIR, "train_oof_logits.npy")))
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values

    nested_preds = np.zeros(len(train_meta), dtype=int)
    chosen_weights = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        best_w, best_wf1 = None, -1
        for w in np.arange(0.0, 1.01, 0.05):
            blend = w * exp09_probs[tr_i] + (1 - w) * exp14_probs[tr_i]
            wf1 = f1_score(true_labels[tr_i], blend.argmax(1), average="weighted", zero_division=0)
            if wf1 > best_wf1:
                best_wf1, best_w = wf1, w
        chosen_weights.append(best_w)

        va_blend = best_w * exp09_probs[va_i] + (1 - best_w) * exp14_probs[va_i]
        nested_preds[va_i] = va_blend.argmax(1)
        print(f"  Fold {fold}: weight chosen on other 4 folds = {best_w:.2f}")

    nested_wf1 = f1_score(true_labels, nested_preds, average="weighted", zero_division=0)
    print(f"\nChosen weights per fold: {chosen_weights}")
    print(f"Honest nested-CV blend WF1: {nested_wf1:.4f}")
    print(f"(vs naive same-set-tuned blend: 0.8088, exp09 alone: 0.8014, exp09 seq-rescored: 0.8078)")


if __name__ == "__main__":
    main()
