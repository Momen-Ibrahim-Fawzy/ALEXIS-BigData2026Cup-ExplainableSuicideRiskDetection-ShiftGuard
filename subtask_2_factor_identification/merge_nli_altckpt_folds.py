"""Combine partial OOF files from train_nli_altckpt_kfold.py runs into the
standalone OOF for Idea 3's alternate checkpoint (khalidalt/DeBERTa-v3-
large-mnli), and report its own standalone honest pseudo-leaderboard score
-- checked BEFORE attempting any ensemble with v20, since a weak standalone
source isn't worth ensembling regardless of diversity (confirmed pattern
this session with factor_all/factor_verified).

Usage:
    python merge_nli_altckpt_folds.py --output-dir outputs_nli_altckpt_kfold
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

    print("\nRunning pseudo-leaderboard honest check (standalone alt-checkpoint model):")
    pseudo_check(os.path.join(output_dir, "oof_probs.npy"), os.path.join(output_dir, "oof_labels.npy"),
                 os.path.join(output_dir, "oof_meta.csv"), label=os.path.basename(output_dir),
                 result_path=os.path.join(output_dir, "pseudo_leaderboard_result.json"))
    print("\nIf this standalone honest score is well below v20's real 0.6156, ensembling with v20 is "
          "unlikely to help regardless of diversity (same pattern seen with factor_all/factor_verified) -- "
          "check before running merge_nli_altckpt_ensemble.py.")


if __name__ == "__main__":
    main()
