"""Ensemble rescored logits from multiple exp09 seed-bag sources (production
seed=42 regenerated read-only + one or more new seeds), softmax-average them,
and gate the combined true-OOF WF1 against production's CONFIRMED number
(0.8078, both the OOF result and the real +0.0030 leaderboard WF1 gain it
produced as v9/v11) before building a submission CSV -- same discipline as
every Sub2 seed-bag idea, adapted to 1a's already-genuine (no per-class
threshold tuning) OOF, so this compares straight OOF WF1 to OOF WF1, no
honest/naive split needed here.

Usage:
    python merge_seedbag_1a.py --dirs ../results_seedbag_1a/production_rescored ../results_seedbag_1a/seed101_rescored \\
        --output-dir ../results_seedbag_1a/ensemble
"""

import argparse, os, sys, json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, ID2RISK, RISK_LEVELS, BASE_DIR

SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
PRODUCTION_WF1 = 0.807789217365434  # confirmed OOF WF1 of the currently-live submission_exp09_..._seqrescored.csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_meta = pd.read_csv(os.path.join(args.dirs[0], "train_meta.csv"))
    lb_meta = pd.read_csv(os.path.join(args.dirs[0], "leaderboard_meta.csv"))
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])

    train_probs, lb_probs = [], []
    for d in args.dirs:
        tl = np.load(os.path.join(d, "train_rescored_logits.npy"))
        ll = np.load(os.path.join(d, "leaderboard_rescored_logits.npy"))
        tp, lp = softmax(tl, axis=1), softmax(ll, axis=1)
        solo_wf1 = f1_score(true_labels, tp.argmax(1), average="weighted", zero_division=0)
        print(f"  {d}: standalone true OOF WF1 = {solo_wf1:.4f}")
        train_probs.append(tp)
        lb_probs.append(lp)

    avg_train = np.mean(train_probs, axis=0)
    avg_lb = np.mean(lb_probs, axis=0)
    combined_wf1 = f1_score(true_labels, avg_train.argmax(1), average="weighted", zero_division=0)
    combined_mf1 = f1_score(true_labels, avg_train.argmax(1), average="macro", zero_division=0)
    print(f"\n{len(args.dirs)}-way seed-bag average, true OOF WF1: {combined_wf1:.4f}  MF1: {combined_mf1:.4f}")
    print(f"Baseline to beat (production, confirmed real +0.0030 WF1 leaderboard gain): {PRODUCTION_WF1:.4f}")

    np.save(os.path.join(args.output_dir, "train_avg_probs.npy"), avg_train)
    np.save(os.path.join(args.output_dir, "leaderboard_avg_probs.npy"), avg_lb)
    result = {"combined_wf1": float(combined_wf1), "combined_mf1": float(combined_mf1),
              "production_wf1": PRODUCTION_WF1, "gain": float(combined_wf1 - PRODUCTION_WF1),
              "sources": args.dirs}
    with open(os.path.join(args.output_dir, "seedbag_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if combined_wf1 <= PRODUCTION_WF1:
        print(f"\n*** Combined WF1 ({combined_wf1:.4f}) did NOT beat production ({PRODUCTION_WF1:.4f}). "
              "Refusing to build a submission. ***")
        sys.exit(1)

    pred_ids = avg_lb.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    submission = pd.DataFrame({"row_id": lb_meta["row_id"], "risk_level": pred_labels})
    print("\nLeaderboard prediction distribution:")
    print(submission["risk_level"].value_counts().to_string())

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_exp09_seedbagged_seqrescored.csv")
    submission.to_csv(out_path, index=False)
    print(f"\n*** PASSED -- saved -> {out_path} ***")


if __name__ == "__main__":
    main()
