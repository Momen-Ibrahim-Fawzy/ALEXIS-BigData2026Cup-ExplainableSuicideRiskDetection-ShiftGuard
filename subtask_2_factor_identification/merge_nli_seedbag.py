"""Combine v20's original 5-fold NLI cross-encoder OOF with 1+ additional
seed-bagged runs (train_nli_seedbag.py), average the probabilities, and
report the honest pseudo-leaderboard score.

Usage:
    python merge_nli_seedbag.py --dirs outputs_nli_finetune_kfold outputs_nli_seedbag_101 outputs_nli_seedbag_202 \\
        --output-dir outputs_nli_seedbag_combined
"""

import argparse, os
import numpy as np
from sklearn.metrics import f1_score

from config import NUM_FACTORS
from pseudo_leaderboard_split import check as pseudo_check


def compute_macro_f1(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return float(np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                     help="e.g. outputs_nli_finetune_kfold outputs_nli_seedbag_101 outputs_nli_seedbag_202")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    labels = np.load(os.path.join(args.dirs[0], "oof_labels.npy"))
    all_probs = []
    for d in args.dirs:
        p = np.load(os.path.join(d, "oof_probs.npy"))
        mf1 = compute_macro_f1(p, labels)
        print(f"  {d}: standalone raw t=0.5 Macro F1 = {mf1:.4f}")
        all_probs.append(p)

    avg_probs = np.mean(all_probs, axis=0)
    avg_mf1 = compute_macro_f1(avg_probs, labels)
    print(f"\n{len(args.dirs)}-way seed-bag average, raw t=0.5: {avg_mf1:.4f}")

    np.save(os.path.join(args.output_dir, "oof_probs.npy"), avg_probs)
    np.save(os.path.join(args.output_dir, "oof_labels.npy"), labels)

    # Reuse the first dir's oof_meta.csv (identical row order/users across all seed runs)
    import shutil
    shutil.copy(os.path.join(args.dirs[0], "oof_meta.csv"), os.path.join(args.output_dir, "oof_meta.csv"))

    print("\nRunning pseudo-leaderboard honest check:")
    pseudo_check(os.path.join(args.output_dir, "oof_probs.npy"), os.path.join(args.output_dir, "oof_labels.npy"),
                 os.path.join(args.output_dir, "oof_meta.csv"), label=os.path.basename(args.output_dir),
                 result_path=os.path.join(args.output_dir, "pseudo_leaderboard_result.json"))

    with open(os.path.join(args.output_dir, "source_dirs.txt"), "w") as f:
        f.write("\n".join(os.path.abspath(d) for d in args.dirs))


if __name__ == "__main__":
    main()
