"""
Build the combined (split-ensemble) OOF and leaderboard probability matrices
for Sub2, converted to logits, with anon_user_id/post_id metadata attached.
Feeds sequence_rescore.py — the Sub2 analogue of Sub1a's user-sequence
re-scoring (backlog item #1).

Usage:
    python extract_seq_data.py --weighted-dir outputs_v8_weighted --vanilla-dir outputs_v8_vanilla
"""

import os, argparse
import numpy as np
import pandas as pd
from config import *
from predict_split_ensemble import load_probs, compute_pos_weights, build_combined

SEQ_DIR = os.path.join(os.path.dirname(__file__), "sequence_data")
os.makedirs(SEQ_DIR, exist_ok=True)

EPS = 1e-6


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted-dir", default=os.path.join(os.path.dirname(__file__), "outputs_v8_weighted"))
    parser.add_argument("--vanilla-dir", default=os.path.join(os.path.dirname(__file__), "outputs_v8_vanilla"))
    args = parser.parse_args()

    w_oof   = load_probs(args.weighted_dir, "oof_probs.npy")
    v_oof   = load_probs(args.vanilla_dir,  "oof_probs.npy")
    labels  = load_probs(args.weighted_dir, "oof_labels.npy")
    w_lb    = load_probs(args.weighted_dir, "leaderboard_probs.npy")
    v_lb    = load_probs(args.vanilla_dir,  "leaderboard_probs.npy")

    pw = compute_pos_weights()
    combined_oof, source = build_combined(w_oof, v_oof, pw)
    combined_lb, _ = build_combined(w_lb, v_lb, pw)

    print(f"Combined OOF shape: {combined_oof.shape}  |  LB shape: {combined_lb.shape}")

    oof_logits = to_logit(combined_oof)
    lb_logits  = to_logit(combined_lb)

    np.save(os.path.join(SEQ_DIR, "train_oof_logits.npy"), oof_logits)
    np.save(os.path.join(SEQ_DIR, "train_oof_labels.npy"), labels)
    np.save(os.path.join(SEQ_DIR, "leaderboard_logits.npy"), lb_logits)

    train_df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    assert len(train_df) == oof_logits.shape[0], "train.xlsx row count must match oof_probs rows"
    assert len(lb_df) == lb_logits.shape[0], "leaderboard.xlsx row count must match leaderboard_probs rows"

    train_df[["row_id", "anon_user_id", "post_id"]].to_csv(
        os.path.join(SEQ_DIR, "train_meta.csv"), index=False)
    lb_df[["row_id", "anon_user_id", "post_id"]].to_csv(
        os.path.join(SEQ_DIR, "leaderboard_meta.csv"), index=False)

    print(f"Saved -> {SEQ_DIR}/")


if __name__ == "__main__":
    main()
