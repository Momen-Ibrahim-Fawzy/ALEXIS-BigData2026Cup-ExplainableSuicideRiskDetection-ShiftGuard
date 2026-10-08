"""Ensemble v20's original NLI cross-encoder OOF with Idea 3's alternate-
checkpoint (khalidalt/DeBERTa-v3-large-mnli) OOF via a weighted average.

The blend weight is chosen by scanning w in [0, 1] and picking whichever
maximizes the MacroF1 on the tune-pool (the same ~133 non-pseudo-leaderboard
users used everywhere else for threshold tuning) -- NOT on the 20-user
pseudo-leaderboard holdout itself, which must stay untouched by any tuning
decision (weight selection counts as one) per pseudo_leaderboard_split.py's
whole rationale. The honest check below is the first and only time the
holdout is touched, exactly as with every other candidate this session.

Usage:
    python merge_nli_altckpt_ensemble.py --v20-dir outputs_nli_finetune_kfold \\
        --alt-dir outputs_nli_altckpt_kfold --output-dir outputs_nli_altckpt_ensemble
"""

import argparse, json, os, shutil
import numpy as np
import pandas as pd

from pseudo_leaderboard_split import check as pseudo_check, get_or_create_split, tune_thresholds, compute_macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20-dir", required=True)
    ap.add_argument("--alt-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    labels = np.load(os.path.join(args.v20_dir, "oof_labels.npy"))
    probs_v20 = np.load(os.path.join(args.v20_dir, "oof_probs.npy"))
    probs_alt = np.load(os.path.join(args.alt_dir, "oof_probs.npy"))
    meta = pd.read_csv(os.path.join(args.v20_dir, "oof_meta.csv"))

    holdout_users = set(get_or_create_split())
    is_holdout = meta["anon_user_id"].isin(holdout_users).values
    tune_idx = ~is_holdout

    best_w, best_score = 1.0, -1.0
    for w in np.arange(0.0, 1.001, 0.05):
        combined = w * probs_v20 + (1 - w) * probs_alt
        th = tune_thresholds(combined[tune_idx], labels[tune_idx])
        score = compute_macro_f1(combined[tune_idx], labels[tune_idx], th)
        if score > best_score:
            best_score, best_w = score, float(w)
    print(f"Best weight on v20 (selected on tune-pool only, holdout untouched): "
          f"w_v20={best_w:.2f}  w_alt={1-best_w:.2f}  (tune-pool MacroF1={best_score:.4f})")
    print(f"  v20 alone (tune-pool):  {compute_macro_f1(probs_v20[tune_idx], labels[tune_idx], tune_thresholds(probs_v20[tune_idx], labels[tune_idx])):.4f}")
    print(f"  alt alone (tune-pool):  {compute_macro_f1(probs_alt[tune_idx], labels[tune_idx], tune_thresholds(probs_alt[tune_idx], labels[tune_idx])):.4f}")

    combined_probs = best_w * probs_v20 + (1 - best_w) * probs_alt
    np.save(os.path.join(args.output_dir, "oof_probs.npy"), combined_probs)
    np.save(os.path.join(args.output_dir, "oof_labels.npy"), labels)
    shutil.copy(os.path.join(args.v20_dir, "oof_meta.csv"), os.path.join(args.output_dir, "oof_meta.csv"))
    with open(os.path.join(args.output_dir, "ensemble_weight.json"), "w") as f:
        json.dump({"weight_v20": best_w, "weight_alt": 1 - best_w}, f, indent=2)

    print("\nRunning pseudo-leaderboard honest check on the fixed-weight ensemble:")
    pseudo_check(os.path.join(args.output_dir, "oof_probs.npy"), os.path.join(args.output_dir, "oof_labels.npy"),
                 os.path.join(args.output_dir, "oof_meta.csv"), label=os.path.basename(args.output_dir),
                 result_path=os.path.join(args.output_dir, "pseudo_leaderboard_result.json"))


if __name__ == "__main__":
    main()
