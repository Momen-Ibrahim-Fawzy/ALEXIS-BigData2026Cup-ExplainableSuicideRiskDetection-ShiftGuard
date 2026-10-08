"""Leaderboard prediction from the already-trained NLI fine-tune
(outputs_nli_finetune/best.pt) -- reuses the model as-is, no retraining.
Rebuilds the identical 107/26/20 user split (deterministic, same SEED) to
recompute the threshold-tuning pool's scores (not saved during training),
then scores the real 378 leaderboard posts.

Usage:
    python predict_nli_finetune.py
"""

import ast, json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import *
from dataset_nli_finetune import build_pairs, NLIPairDataset
from model_nli_finetune import NLIFactorClassifier, NLI_MODEL
from pseudo_leaderboard_split import get_or_create_split
from train_nli_finetune import run_epoch, pairs_to_matrix, tune_thresholds, compute_macro_f1, SEED, MAX_LENGTH, BATCH_SIZE

from transformers import AutoTokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_nli_finetune")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)

    model = NLIFactorClassifier().to(device).float()
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt"), weights_only=False, map_location=device)["model"])
    model.eval()

    # Rebuild the identical split used during training (deterministic, same SEED)
    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["factors_list"] = df["factors"].apply(lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])
    pseudo_lb_users = set(get_or_create_split())
    is_pseudo_lb = df["anon_user_id"].isin(pseudo_lb_users).values
    remaining_df = df[~is_pseudo_lb].reset_index(drop=False)
    remaining_users = sorted(remaining_df["anon_user_id"].unique())
    rng = np.random.RandomState(SEED)
    rng.shuffle(remaining_users)
    n_thresh_users = max(1, int(0.2 * len(remaining_users)))
    thresh_users = set(remaining_users[:n_thresh_users])
    thresh_df = remaining_df[remaining_df["anon_user_id"].isin(thresh_users)].reset_index(drop=True)

    print(f"Recomputing thresholds on {len(thresh_df)} threshold-tuning-pool posts...")
    thresh_pairs = build_pairs(thresh_df)
    thresh_ds = NLIPairDataset(thresh_pairs, tokenizer, MAX_LENGTH)
    thresh_loader = DataLoader(thresh_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    _, scores, labels_flat, row_idx, factor_idx = run_epoch(model, thresh_loader, None, None, device, train=False)
    thresh_probs = pairs_to_matrix(scores, row_idx, factor_idx, len(thresh_df))
    thresh_labels = pairs_to_matrix(labels_flat, row_idx, factor_idx, len(thresh_df))
    thresholds = tune_thresholds(thresh_probs, thresh_labels)

    with open(os.path.join(OUT_DIR, "factor_thresholds_nli_finetune.json"), "w") as f:
        json.dump({FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)}, f, indent=2)

    # ---- Leaderboard inference ----
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    print(f"Scoring {len(lb_df)} leaderboard posts x {NUM_FACTORS} factors = {len(lb_df)*NUM_FACTORS} pairs...")

    # Leaderboard posts have no factors_list -- build_pairs needs SOME label column;
    # fill with empty lists (labels unused for prediction, only for the pair-building loop shape).
    lb_df_for_pairs = lb_df.copy()
    lb_df_for_pairs["factors_list"] = [[] for _ in range(len(lb_df))]
    lb_pairs = build_pairs(lb_df_for_pairs)
    lb_ds = NLIPairDataset(lb_pairs, tokenizer, MAX_LENGTH)
    lb_loader = DataLoader(lb_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    _, lb_scores, _, lb_row_idx, lb_factor_idx = run_epoch(model, lb_loader, None, None, device, train=False)
    lb_probs = pairs_to_matrix(lb_scores, lb_row_idx, lb_factor_idx, len(lb_df))
    np.save(os.path.join(OUT_DIR, "leaderboard_probs.npy"), lb_probs)

    preds = (lb_probs >= thresholds).astype(int)
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"], "factors": str(factors) if factors else "[]"})
    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "factor_predictions_nli_finetune.csv")
    out.to_csv(out_path, index=False)
    n = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"\nSaved -> {out_path}  (mean factors/post: {n.mean():.2f}, zero: {(n==0).sum()}/{len(out)})")


if __name__ == "__main__":
    main()
