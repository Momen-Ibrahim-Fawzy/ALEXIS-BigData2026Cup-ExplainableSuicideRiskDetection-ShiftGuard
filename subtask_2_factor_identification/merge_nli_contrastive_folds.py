"""Combine partial OOF files from one or more train_nli_contrastive_kfold.py
runs into the final oof_probs.npy, and report the honest pseudo-leaderboard
score for Idea 2 (contrastive learning, isolated from the failed "v2"
combination).

Usage:
    python merge_nli_contrastive_folds.py --output-dir outputs_nli_contrastive_kfold
"""

import argparse, glob, os
import numpy as np
from sklearn.metrics import f1_score

from config import NUM_FACTORS
from pseudo_leaderboard_split import check as pseudo_check


def compute_macro_f1(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return float(np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    output_dir = os.path.abspath(args.output_dir)

    partial_probs_paths = sorted(glob.glob(os.path.join(output_dir, "oof_probs_folds_*.npy")))
    partial_filled_paths = sorted(glob.glob(os.path.join(output_dir, "oof_filled_folds_*.npy")))
    if not partial_probs_paths:
        raise FileNotFoundError(f"No oof_probs_folds_*.npy found in {output_dir}")
    print(f"Merging {len(partial_probs_paths)} partial file(s): {[os.path.basename(p) for p in partial_probs_paths]}")

    oof_probs, oof_filled = None, None
    for pp, fp in zip(partial_probs_paths, partial_filled_paths):
        p, f = np.load(pp), np.load(fp)
        if oof_probs is None:
            oof_probs = np.zeros_like(p)
            oof_filled = np.zeros_like(f)
        overlap = oof_filled & f
        if overlap.any():
            raise ValueError(f"{overlap.sum()} rows filled by more than one partial file -- refusing to merge.")
        oof_probs[f] = p[f]
        oof_filled |= f

    if not oof_filled.all():
        raise ValueError(f"{(~oof_filled).sum()} rows never filled -- check --fold-ids coverage across runs.")

    oof_labels = np.load(os.path.join(output_dir, "oof_labels.npy"))
    np.save(os.path.join(output_dir, "oof_probs.npy"), oof_probs)

    mf1 = compute_macro_f1(oof_probs, oof_labels)
    print(f"\nTrue 5-fold OOF Macro F1 (t=0.5): {mf1:.4f}")
    print("(Compare to v20's real leaderboard result 0.6156 -- the bar Idea 2 must beat honestly to submit)")

    print("\nRunning pseudo-leaderboard honest check:")
    pseudo_check(os.path.join(output_dir, "oof_probs.npy"), os.path.join(output_dir, "oof_labels.npy"),
                 os.path.join(output_dir, "oof_meta.csv"), label=os.path.basename(output_dir),
                 result_path=os.path.join(output_dir, "pseudo_leaderboard_result.json"))


if __name__ == "__main__":
    main()
