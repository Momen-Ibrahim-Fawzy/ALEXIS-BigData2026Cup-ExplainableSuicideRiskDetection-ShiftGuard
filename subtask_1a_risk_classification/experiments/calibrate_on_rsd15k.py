"""Post-hoc calibration correction for exp09, fit on RSD-15K.

Motivation: exp09 shows a specific, consistent error signature on RSD-15K
(external, labeled, 14,613 posts, zero user overlap) -- it over-predicts the
severe classes (Behavior/Attempt: high recall, low precision) and
under-catches the milder ones (Indicator/Ideation: high precision, low
recall). This fits a correction directly targeting that signature, using
RSD-15K's own labels as calibration ground truth -- no retraining of the
backbone at all, purely a post-processing step on exp09's existing output
probabilities.

Two variants tried:
  1. Bias-only: learn a 4-dim additive correction to log-probs (4 free
     params) -- low overfitting risk.
  2. Matrix (Dirichlet-style) scaling: learn a full 4x4 W + 4 b on log-probs
     (20 free params) -- more flexible, more overfitting risk.

Both are fit on one half of RSD-15K (split by user, not by row, to avoid
leakage) and validated on the other half -- genuine held-out signal, not the
same data used for fitting. The same correction is then checked against
exp09's own true-OOF WF1 on the competition's training data, to make sure it
doesn't damage the thing it's already good at.

Usage:
    python calibrate_on_rsd15k.py
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, ID2RISK, DATA_DIR, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
EPS = 1e-8


def load_rsd():
    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    probs = np.load(os.path.join(RESULTS_DIR, "exp09_rsd15k_probs.npy"))
    labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])
    return rsd, probs, labels


def fit_bias_only(logp_fit, y_fit, n_classes=4):
    """Learn a 4-dim additive bias on log-probs via multinomial logistic
    regression with the log-probs themselves as a fixed offset (this is
    exactly bias-corrected softmax: softmax(log p + b))."""
    # Closed-form: the optimal additive bias for a fixed base classifier is
    # the one that matches class marginals (a standard "prior correction" /
    # label-shift correction, Saerens et al. 2002):
    # fit b_c = log(empirical_class_freq_fit) -
    # log(mean predicted_prob_fit_for_class_c) -- a standard "prior
    # correction" / label-shift correction (Saerens et al. 2002).
    empirical = np.bincount(y_fit, minlength=n_classes) / len(y_fit)
    mean_pred = np.exp(logp_fit).mean(axis=0)
    bias = np.log(empirical + EPS) - np.log(mean_pred + EPS)
    return bias


def apply_bias(probs, bias):
    logp = np.log(probs + EPS) + bias
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def fit_matrix_scaling(logp_fit, y_fit):
    """Dirichlet-calibration-style: logistic regression with log-probs as
    features, learns a full 4x4 weight matrix + 4 biases."""
    clf = LogisticRegression(max_iter=3000, C=1.0)
    clf.fit(logp_fit, y_fit)
    return clf


def apply_matrix(probs, clf):
    logp = np.log(probs + EPS)
    return clf.predict_proba(logp)


def evaluate(name, probs, labels):
    preds = probs.argmax(axis=1)
    wf1 = f1_score(labels, preds, average="weighted", zero_division=0)
    print(f"  {name}: WF1={wf1:.4f}")
    return wf1


def main():
    rsd, probs, labels = load_rsd()
    groups = rsd["users"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    fit_idx, hold_idx = next(gss.split(rsd, groups=groups))
    print(f"RSD-15K split: {len(fit_idx)} fit rows / {len(hold_idx)} holdout rows "
          f"(user-grouped, no overlap)")

    probs_fit, y_fit = probs[fit_idx], labels[fit_idx]
    probs_hold, y_hold = probs[hold_idx], labels[hold_idx]
    logp_fit = np.log(probs_fit + EPS)

    print(f"\n{'='*70}\n  BASELINE (uncorrected exp09) on RSD-15K holdout\n{'='*70}")
    evaluate("uncorrected", probs_hold, y_hold)

    print(f"\n{'='*70}\n  Variant 1: bias-only (prior/label-shift correction)\n{'='*70}")
    bias = fit_bias_only(logp_fit, y_fit)
    print(f"  learned bias (indicator, ideation, behavior, attempt): {bias.round(3)}")
    corrected_hold_bias = apply_bias(probs_hold, bias)
    evaluate("bias-corrected", corrected_hold_bias, y_hold)

    print(f"\n{'='*70}\n  Variant 2: matrix (Dirichlet-style) scaling\n{'='*70}")
    clf = fit_matrix_scaling(logp_fit, y_fit)
    corrected_hold_matrix = apply_matrix(probs_hold, clf)
    evaluate("matrix-corrected", corrected_hold_matrix, y_hold)

    print(f"\n{'='*70}\n  Classification report: uncorrected vs bias-corrected (RSD holdout)\n{'='*70}")
    names = ["indicator", "ideation", "behavior", "attempt"]
    print("-- uncorrected --")
    print(classification_report(y_hold, probs_hold.argmax(1), target_names=names, zero_division=0))
    print("-- bias-corrected --")
    print(classification_report(y_hold, corrected_hold_bias.argmax(1), target_names=names, zero_division=0))

    # Sanity check: does the RSD-fit correction help or hurt on the
    # competition's OWN true-OOF (train.xlsx)? Refit bias using the FULL
    # RSD-15K (more stable), since this is now a one-shot deployment check,
    # not held-out validation of the correction's RSD-generalization itself.
    print(f"\n{'='*70}\n  Sanity check: same correction applied to exp09's own true-OOF (train.xlsx)\n{'='*70}")
    full_bias = fit_bias_only(np.log(probs + EPS), labels)
    print(f"  full-RSD-15K bias: {full_bias.round(3)}")

    train_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    train_probs = np.exp(train_logits) / np.exp(train_logits).sum(axis=1, keepdims=True)
    train_df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    train_df["risk_norm"] = train_df["suicide risk"].str.strip().str.lower()
    train_labels = np.array([RISK2ID[r] for r in train_df["risk_norm"]])

    evaluate("exp09 true-OOF, uncorrected", train_probs, train_labels)
    corrected_train = apply_bias(train_probs, full_bias)
    evaluate("exp09 true-OOF, RSD-bias-corrected", corrected_train, train_labels)

    # Save the fitted full-data bias + a preview of what it would do to the
    # leaderboard predictions, for use if this clears the bar.
    np.save(os.path.join(RESULTS_DIR, "rsd15k_bias_correction.npy"), full_bias)
    lb_logits = np.load(os.path.join(SEQ_DIR, "leaderboard_logits.npy"))
    lb_probs = np.exp(lb_logits) / np.exp(lb_logits).sum(axis=1, keepdims=True)
    lb_corrected = apply_bias(lb_probs, full_bias)
    orig_labels = [ID2RISK[i].capitalize() for i in lb_probs.argmax(1)]
    new_labels = [ID2RISK[i].capitalize() for i in lb_corrected.argmax(1)]
    n_changed = sum(a != b for a, b in zip(orig_labels, new_labels))
    print(f"\n  Leaderboard preview: correction would change {n_changed}/378 predictions")
    print(f"  original distribution: {pd.Series(orig_labels).value_counts().to_dict()}")
    print(f"  corrected distribution: {pd.Series(new_labels).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
