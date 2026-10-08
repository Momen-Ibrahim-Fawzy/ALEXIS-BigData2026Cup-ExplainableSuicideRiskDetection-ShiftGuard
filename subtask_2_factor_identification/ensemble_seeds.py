"""Average OOF (and leaderboard, if available) probabilities across multiple
runs of the *same* oversample-source + loss config, trained with different
--seed values only (GroupKFold splits are identical across seeds since they
depend only on anon_user_id, not randomness -- so this is a clean apples-to-
apples bagging comparison, not a different validation partition).

This is the lowest-risk lever available for Sub2 right now: same
architecture, same data, same loss -- only the random init/dropout/batch
order differs. Contrast with the already-tried (and reverted) idea of
blending in a *different* oversample source, which looked good on OOF here
too but sometimes doesn't survive contact with the real leaderboard (see
BACKLOG.md's v13 3-way-ensemble entry, and the exp09+exp14 Sub1a episode).

Thresholds are re-tuned ONCE on the seed-averaged OOF matrix (single
principled pass, matching v12's discipline) -- not per-seed, to avoid the
small-sample threshold-overfitting that sank the Sub1a bias-calibration
attempt and an earlier Sub2 sequence-rescoring attempt.

Usage:
    python ensemble_seeds.py --dirs outputs_factor_verified_weighted_seed42 \\
        outputs_factor_verified_weighted_seed101 outputs_factor_verified_weighted_seed202
"""

import argparse, json, os
import numpy as np
from sklearn.metrics import f1_score

from config import *


def compute_macro_f1(probs, labels, thresholds):
    preds = (probs >= thresholds).astype(int)
    return np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0)
                     for i in range(NUM_FACTORS)])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                     help="2+ output dirs from separate --seed runs of the same config")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    oof_probs_list = [np.load(os.path.join(d, "oof_probs.npy")) for d in args.dirs]
    labels = np.load(os.path.join(args.dirs[0], "oof_labels.npy"))

    t05 = np.full(NUM_FACTORS, 0.5)
    single_run_tuned = []
    for d, p in zip(args.dirs, oof_probs_list):
        mf1_05 = compute_macro_f1(p, labels, t05)
        t = tune_thresholds(p, labels)
        mf1_tuned = compute_macro_f1(p, labels, t)
        single_run_tuned.append(mf1_tuned)
        print(f"  {d}: t=0.5 -> {mf1_05:.4f}  |  tuned -> {mf1_tuned:.4f}")

    avg_probs = np.mean(oof_probs_list, axis=0)
    print(f"\n{len(args.dirs)}-way seed-bag average, t=0.5: "
          f"{compute_macro_f1(avg_probs, labels, t05):.4f}")

    thresholds = tune_thresholds(avg_probs, labels)
    tuned_mf1 = compute_macro_f1(avg_probs, labels, thresholds)
    print(f"{len(args.dirs)}-way seed-bag average, tuned thresholds: {tuned_mf1:.4f}")
    print(f"(compare to best single-seed tuned: {max(single_run_tuned):.4f}, "
          f"mean single-seed tuned: {np.mean(single_run_tuned):.4f})")

    out_dir = args.out_dir or (args.dirs[0].rstrip("/") + "_seedbag")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "oof_probs.npy"), avg_probs)
    np.save(os.path.join(out_dir, "oof_labels.npy"), labels)
    with open(os.path.join(out_dir, "factor_thresholds.json"), "w") as f:
        json.dump({"thresholds": {FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)},
                    "tuned_macro_f1": float(tuned_mf1)}, f, indent=2)

    lb_paths = [os.path.join(d, "leaderboard_probs.npy") for d in args.dirs]
    if all(os.path.exists(p) for p in lb_paths):
        lb_avg = np.mean([np.load(p) for p in lb_paths], axis=0)
        np.save(os.path.join(out_dir, "leaderboard_probs.npy"), lb_avg)
        print(f"Saved averaged leaderboard probs -> {out_dir}/leaderboard_probs.npy")
    else:
        print("Leaderboard probs missing for some seeds -- run "
              "`python predict.py --output-dir <seed_dir>` for each seed dir first "
              "if you want this bagged source pluggable into a submission.")

    print(f"\nSaved seed-bagged OOF + thresholds -> {out_dir}/")
    if tuned_mf1 <= max(single_run_tuned):
        print("\n*** Seed-bagging did NOT beat the best single seed on tuned true "
              "OOF. Do not use this as a submission source. ***")


if __name__ == "__main__":
    main()
