"""Combine partial OOF files from one or more train_v2.py runs (e.g. two
parallel GPU processes each covering a disjoint --fold-ids subset) into the
final oof_probs.npy / oof_labels.npy, and report the true 5-fold OOF Macro
F1 plus the pseudo-leaderboard honest check.

Usage:
    python merge_v2_folds.py --output-dir outputs_v2_factor_verified
"""

import argparse, glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import *
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

    oof_probs = None
    oof_filled = None
    for pp, fp in zip(partial_probs_paths, partial_filled_paths):
        p = np.load(pp)
        f = np.load(fp)
        if oof_probs is None:
            oof_probs = np.zeros_like(p)
            oof_filled = np.zeros_like(f)
        overlap = oof_filled & f
        if overlap.any():
            raise ValueError(f"{overlap.sum()} rows filled by more than one partial file -- "
                              f"fold subsets were not actually disjoint, refusing to merge.")
        oof_probs[f] = p[f]
        oof_filled |= f

    if not oof_filled.all():
        missing = (~oof_filled).sum()
        raise ValueError(f"{missing} rows never filled by any fold -- did you train all 5 folds "
                          f"(across however many parallel processes)? Check --fold-ids coverage.")

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True) if "anon_user_id" in df.columns else df
    import ast
    orig_df["factors_list"] = orig_df["factors"].apply(
        lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])
    oof_labels = np.zeros((len(orig_df), NUM_FACTORS), dtype=np.float32)
    for i, row in orig_df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                oof_labels[i, FACTOR2ID[fl]] = 1.0

    np.save(os.path.join(output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(output_dir, "oof_labels.npy"), oof_labels)
    orig_df[["row_id", "anon_user_id"]].to_csv(os.path.join(output_dir, "oof_meta.csv"), index=False)

    mf1 = compute_macro_f1(oof_probs, oof_labels)
    print(f"\nTrue 5-fold OOF Macro F1 (t=0.5, naive -- matches how other baselines are usually first reported): {mf1:.4f}")
    print(f"(compare to: factor_verified 0.5806, factor_all 0.5785, current best 2-way ensemble 0.5982 -- "
          f"all THRESHOLD-TUNED numbers, this one is raw t=0.5; run predict_v2.py for a tuned comparison)")

    print("\nRunning pseudo-leaderboard honest check (per BACKLOG.md, this is the number to actually trust "
          "-- v18's naive OOF said +0.0083 and the real leaderboard showed -0.0326; the pseudo-leaderboard "
          "check alone predicted the real result to within 0.004):")
    pseudo_check(os.path.join(output_dir, "oof_probs.npy"), os.path.join(output_dir, "oof_labels.npy"),
                 os.path.join(output_dir, "oof_meta.csv"), label=os.path.basename(output_dir),
                 result_path=os.path.join(output_dir, "pseudo_leaderboard_result.json"))


if __name__ == "__main__":
    main()
