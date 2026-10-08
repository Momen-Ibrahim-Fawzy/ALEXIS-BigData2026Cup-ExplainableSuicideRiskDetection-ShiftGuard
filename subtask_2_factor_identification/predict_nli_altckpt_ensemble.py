"""Leaderboard prediction for Idea 3's v20 + alt-checkpoint ensemble.
Unlike the seed-bag ensemble (Idea 1), the two sources use DIFFERENT
tokenizers/checkpoints, so the leaderboard set is tokenized and scored
separately for each source, then combined with the fixed weight chosen by
merge_nli_altckpt_ensemble.py (ensemble_weight.json).

Usage:
    python predict_nli_altckpt_ensemble.py --v20-dir outputs_nli_finetune_kfold \\
        --alt-dir outputs_nli_altckpt_kfold --output-dir outputs_nli_altckpt_ensemble
"""

import argparse, json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from config import *
from dataset_nli_finetune import build_pairs, NLIPairDataset
from model_nli_finetune import NLIFactorClassifier, NLI_MODEL
from train_nli_finetune import run_epoch, pairs_to_matrix
from train_nli_altckpt_kfold import ALT_NLI_MODEL
from pseudo_leaderboard_split import get_or_create_split

MAX_LENGTH = 256
BATCH_SIZE = 16


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


def ensemble_source_probs(model_name, fold_dir, lb_df, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lb_df_for_pairs = lb_df.copy()
    lb_df_for_pairs["factors_list"] = [[] for _ in range(len(lb_df))]
    lb_pairs = build_pairs(lb_df_for_pairs)
    lb_ds = NLIPairDataset(lb_pairs, tokenizer, MAX_LENGTH)
    lb_loader = DataLoader(lb_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model_paths = [os.path.join(fold_dir, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS) if os.path.exists(os.path.join(fold_dir, f"fold_{f}", "best.pt"))]
    print(f"  Ensembling {len(model_paths)} fold models from {fold_dir} (checkpoint: {model_name})")

    all_probs = []
    for p in model_paths:
        m = NLIFactorClassifier(model_name=model_name).to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        _, scores, _, row_idx, factor_idx = run_epoch(m, lb_loader, None, None, device, train=False)
        probs = pairs_to_matrix(scores, row_idx, factor_idx, len(lb_df))
        all_probs.append(probs)
        del m
        torch.cuda.empty_cache()
    return np.mean(all_probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20-dir", required=True)
    ap.add_argument("--alt-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    v20_dir = os.path.abspath(args.v20_dir)
    alt_dir = os.path.abspath(args.alt_dir)
    output_dir = os.path.abspath(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(output_dir, "ensemble_weight.json")) as f:
        weights = json.load(f)
    w_v20 = weights["weight_v20"]
    print(f"Ensemble weight: v20={w_v20:.2f}  alt={1-w_v20:.2f}")

    oof_probs = np.load(os.path.join(output_dir, "oof_probs.npy"))
    oof_labels = np.load(os.path.join(output_dir, "oof_labels.npy"))
    oof_meta = pd.read_csv(os.path.join(output_dir, "oof_meta.csv"))
    holdout_users = set(get_or_create_split())
    is_holdout = oof_meta["anon_user_id"].isin(holdout_users).values
    thresholds = tune_thresholds(oof_probs[~is_holdout], oof_labels[~is_holdout])
    with open(os.path.join(output_dir, "factor_thresholds_nli_altckpt_ensemble.json"), "w") as f:
        json.dump({FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)}, f, indent=2)

    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")

    print("\nScoring leaderboard with v20's checkpoint:")
    probs_v20 = ensemble_source_probs(NLI_MODEL, v20_dir, lb_df, device)
    print("\nScoring leaderboard with alt checkpoint:")
    probs_alt = ensemble_source_probs(ALT_NLI_MODEL, alt_dir, lb_df, device)

    avg = w_v20 * probs_v20 + (1 - w_v20) * probs_alt
    np.save(os.path.join(output_dir, "leaderboard_probs.npy"), avg)

    preds = (avg >= thresholds).astype(int)
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"], "factors": str(factors) if factors else "[]"})
    out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "factor_predictions_nli_altckpt_ensemble.csv")
    out.to_csv(out_path, index=False)
    n = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"Saved -> {out_path}  (mean factors/post: {n.mean():.2f}, zero: {(n==0).sum()}/{len(out)})")


if __name__ == "__main__":
    main()
