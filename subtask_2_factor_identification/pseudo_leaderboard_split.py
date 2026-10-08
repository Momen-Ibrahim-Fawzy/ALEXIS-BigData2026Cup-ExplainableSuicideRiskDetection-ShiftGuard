"""Fixed, reusable pseudo-leaderboard user split for Sub2.

v18 (seed-bagged 2-way ensemble) showed a 0.041 gap between GroupKFold OOF
(+0.0083 predicted) and the real leaderboard (-0.0326 actual) -- the worst
OOF-to-real gap this project has seen, despite the technique being
structurally the "safest" kind (same architecture/data, only init differs).
GroupKFold OOF only measures generalization to unseen TRAIN users; it cannot
detect whatever train-vs-leaderboard population difference is actually
driving these repeated failures (the adversarial-validation check found
AUC~0.69 between train and leaderboard text).

This holds out a FIXED set of ~20 users (never used for ANY tuning decision)
as a standing, reusable "pseudo-leaderboard" -- not a new training run.
Since GroupKFold already produces true OOF predictions for every row (each
fold's own held-out checkpoint scores rows it never trained on), the pseudo-
leaderboard users' OOF predictions are ALREADY honestly out-of-fold. All this
script does is: (1) fix which users count as the pseudo-leaderboard, (2) tune
thresholds ONLY on the other ~133 users' OOF, (3) score those thresholds on
the pseudo-leaderboard users' OOF -- a leakage-free estimate, at zero extra
training cost, exactly analogous to nested-CV but against a fixed external
partition instead of recycling the same folds.

Usage:
    python pseudo_leaderboard_split.py --check <oof_probs.npy> <oof_labels.npy> <meta_csv_with_anon_user_id>
"""

import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import *

SPLIT_PATH = os.path.join(os.path.dirname(__file__), "pseudo_leaderboard_users.json")
N_HOLDOUT_USERS = 20
SPLIT_SEED = 42


def get_or_create_split():
    """Fixed once, reused forever -- so every experiment is checked against
    the exact same held-out users, making results comparable across runs."""
    if os.path.exists(SPLIT_PATH):
        with open(SPLIT_PATH) as f:
            return json.load(f)["pseudo_leaderboard_users"]

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    users = sorted(df["anon_user_id"].unique().tolist())
    rng = np.random.RandomState(SPLIT_SEED)
    holdout = sorted(rng.choice(users, size=N_HOLDOUT_USERS, replace=False).tolist())
    with open(SPLIT_PATH, "w") as f:
        json.dump({"pseudo_leaderboard_users": holdout, "seed": SPLIT_SEED,
                    "n_total_users": len(users)}, f, indent=2)
    print(f"  Created new pseudo-leaderboard split: {len(holdout)}/{len(users)} users -> {SPLIT_PATH}")
    return holdout


def tune_thresholds(probs, labels):
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    for i in range(NUM_FACTORS):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def compute_macro_f1(probs, labels, thresholds):
    preds = (probs >= thresholds).astype(int)
    return np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)])


def check(oof_probs_path, oof_labels_path, meta_csv_path, label="experiment", result_path=None):
    """The core diagnostic: tune on ~133 users, score on the 20 held-out ones."""
    holdout_users = set(get_or_create_split())

    probs = np.load(oof_probs_path)
    labels = np.load(oof_labels_path)
    meta = pd.read_csv(meta_csv_path) if meta_csv_path.endswith(".csv") \
        else pd.read_excel(meta_csv_path, engine="openpyxl")
    assert len(meta) == len(probs), f"meta ({len(meta)}) vs probs ({len(probs)}) length mismatch"

    is_holdout = meta["anon_user_id"].isin(holdout_users).values
    tune_idx, holdout_idx = ~is_holdout, is_holdout
    print(f"  [{label}] tuning pool: {tune_idx.sum()} rows  |  pseudo-leaderboard: {holdout_idx.sum()} rows")

    thresholds = tune_thresholds(probs[tune_idx], labels[tune_idx])
    honest_score = compute_macro_f1(probs[holdout_idx], labels[holdout_idx], thresholds)

    naive_thresholds = tune_thresholds(probs, labels)
    naive_score = compute_macro_f1(probs, labels, naive_thresholds)

    print(f"  [{label}] naive (tune+score on all 153 users, matches historical baselines): {naive_score:.4f}")
    print(f"  [{label}] HONEST (tune on 133, score on 20 never-touched)                  : {honest_score:.4f}")
    print(f"  [{label}] gap: {honest_score - naive_score:+.4f}")

    if result_path:
        with open(result_path, "w") as f:
            json.dump({"label": label, "naive_score": float(naive_score),
                        "honest_score": float(honest_score),
                        "gap": float(honest_score - naive_score)}, f, indent=2)
        print(f"  [{label}] saved -> {result_path}")

    return naive_score, honest_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs=3, metavar=("OOF_PROBS", "OOF_LABELS", "META"),
                     help="Run the honest-vs-naive check on a saved OOF array")
    ap.add_argument("--label", default="experiment")
    ap.add_argument("--result-path", default=None)
    args = ap.parse_args()
    if args.check:
        check(*args.check, label=args.label, result_path=args.result_path)
    else:
        get_or_create_split()
