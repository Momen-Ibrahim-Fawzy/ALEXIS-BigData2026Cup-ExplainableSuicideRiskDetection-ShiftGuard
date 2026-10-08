"""Subtask 2: Per-factor threshold tuning using OOF probabilities.

Run AFTER train.py completes all folds. Uses saved OOF probabilities.
"""

import os, json, argparse
import numpy as np
from sklearn.metrics import f1_score
from config import *


def tune(output_dir=None):
    if output_dir:
        globals()["OUTPUT_DIR"] = os.path.abspath(output_dir)

    probs = np.load(os.path.join(OUTPUT_DIR, "oof_probs.npy"))
    labels = np.load(os.path.join(OUTPUT_DIR, "oof_labels.npy"))
    print(f"OOF shape: probs={probs.shape}, labels={labels.shape}")

    # Default Macro F1
    default_preds = (probs >= 0.5).astype(int)
    default_f1s = [f1_score(labels[:,i], default_preds[:,i], zero_division=0) for i in range(NUM_FACTORS)]
    default_macro = np.mean(default_f1s)
    print(f"\nDefault (t=0.5) Macro F1: {default_macro:.4f}")

    # Per-factor sweep
    thresholds = np.full(NUM_FACTORS, 0.5)
    tuned_f1s = []
    candidates = np.arange(0.05, 0.95, 0.025)

    print(f"\n{'Factor':<48s} {'Default':>7} {'Tuned':>7} {'Thresh':>6} {'Support':>7}")
    print("-" * 80)

    for i in range(NUM_FACTORS):
        best_f1, best_t = 0, 0.5
        for t in candidates:
            f1 = f1_score(labels[:,i], (probs[:,i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[i] = best_t
        tuned_f1s.append(best_f1)
        support = int(labels[:,i].sum())
        print(f"  {FACTOR_LIST[i]:<46s} {default_f1s[i]:>7.3f} {best_f1:>7.3f} {best_t:>6.3f} {support:>7d}")

    tuned_macro = np.mean(tuned_f1s)
    improvement = tuned_macro - default_macro

    print(f"\n{'='*60}")
    print(f"  Default Macro F1: {default_macro:.4f}")
    print(f"  Tuned Macro F1:   {tuned_macro:.4f}  (+{improvement:.4f})")
    print(f"{'='*60}")

    # Save
    result = {
        "default_macro_f1": float(default_macro),
        "tuned_macro_f1": float(tuned_macro),
        "improvement": float(improvement),
        "thresholds": {FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)},
        "per_factor_f1": {FACTOR_LIST[i]: float(tuned_f1s[i]) for i in range(NUM_FACTORS)},
    }
    path = os.path.join(OUTPUT_DIR, "factor_thresholds.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: outputs/)")
    args = parser.parse_args()
    tune(output_dir=args.output_dir)
