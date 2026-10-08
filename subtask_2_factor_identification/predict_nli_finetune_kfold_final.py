"""
Runs v37's actual Sub2 factor-identification model (outputs_nli_finetune_kfold,
5-fold NLI cross-encoder ensemble) on a custom input file instead of the fixed
leaderboard.xlsx. Reuses the existing checkpoints and already-tuned thresholds
verbatim; writes to new filenames so nothing in outputs_nli_finetune_kfold/ is
overwritten.

Usage:
    python predict_nli_finetune_kfold_final.py --input /path/to/final_prediction_input.xlsx \
        --output-dir outputs_nli_finetune_kfold --out-name final_prediction_subtask2_factors.csv
"""

import argparse, json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import *
from dataset_nli_finetune import build_pairs, NLIPairDataset
from model_nli_finetune import NLIFactorClassifier, NLI_MODEL
from train_nli_finetune import run_epoch, pairs_to_matrix

MAX_LENGTH = 256
BATCH_SIZE = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(output_dir, "factor_thresholds_nli_finetune_kfold.json")) as f:
        thr_map = json.load(f)
    thresholds = np.array([thr_map[f] for f in FACTOR_LIST])

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    df = pd.read_excel(args.input, engine="openpyxl")
    print(f"Loaded {len(df)} rows from {args.input}")
    df_for_pairs = df.copy()
    df_for_pairs["factors_list"] = [[] for _ in range(len(df))]
    pairs = build_pairs(df_for_pairs)
    ds = NLIPairDataset(pairs, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model_paths = [os.path.join(output_dir, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS) if os.path.exists(os.path.join(output_dir, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} fold models")

    all_probs = []
    for p in model_paths:
        print(f"  <- {p}")
        m = NLIFactorClassifier().to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        _, scores, _, row_idx, factor_idx = run_epoch(m, loader, None, None, device, train=False)
        probs = pairs_to_matrix(scores, row_idx, factor_idx, len(df))
        all_probs.append(probs)
        del m
        torch.cuda.empty_cache()

    avg = np.mean(all_probs, axis=0)
    np.save(os.path.join(output_dir, args.out_name.replace(".csv", "_probs.npy")), avg)

    preds = (avg >= thresholds).astype(int)
    rows = []
    for i in range(len(df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": df.iloc[i]["row_id"], "factors": str(factors) if factors else "[]"})
    out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, args.out_name)
    out.to_csv(out_path, index=False)
    n = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"Saved -> {out_path}  (mean factors/post: {n.mean():.2f}, zero: {(n==0).sum()}/{len(out)})")


if __name__ == "__main__":
    main()
