"""Generate leaderboard risk predictions from the trained ordinal
cumulative-NLI 5-fold ensemble.

Usage:
    python predict_nli_ordinal_kfold.py --output-dir outputs_nli_ordinal
"""

import argparse, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.xlsx")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

SUB2_DIR = os.path.join(BASELINES_DIR, "subtask_2_factor_identification")
SUB2_AUX_LB_PATH = os.path.join(SUB2_DIR, "outputs_nli_moreepochs_seedbag_combined", "leaderboard_probs.npy")

from risk_hypotheses import RISK_LEVELS, decode_ordinal
from dataset_nli_ordinal import build_pairs, NLIOrdinalDataset, pairs_to_matrix
from model_nli_ordinal import NLIOrdinalRiskClassifier, NLI_MODEL, NUM_FACTORS
from train_nli_ordinal_kfold import run_epoch, MAX_LENGTH, BATCH_SIZE, N_FOLDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    if not os.path.exists(SUB2_AUX_LB_PATH):
        raise FileNotFoundError(
            f"\n\nMissing Sub2 auxiliary factor-probability file: {SUB2_AUX_LB_PATH}\n"
            f"Restore outputs_nli_moreepochs_seedbag_combined/leaderboard_probs.npy "
            f"(and oof_probs.npy, needed by training) to run this.")
    aux_lb = np.load(SUB2_AUX_LB_PATH).astype(np.float32)
    assert aux_lb.shape == (len(lb_df), NUM_FACTORS), \
        f"Sub2 aux leaderboard shape {aux_lb.shape} != expected ({len(lb_df)}, {NUM_FACTORS})"

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    lb_pairs = build_pairs(lb_df, global_idx_col=None)
    lb_ds = NLIOrdinalDataset(lb_pairs, tokenizer, MAX_LENGTH, aux_lb)
    lb_loader = DataLoader(lb_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model_paths = [os.path.join(output_dir, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS) if os.path.exists(os.path.join(output_dir, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} fold models")

    all_probs = []
    for p in model_paths:
        m = NLIOrdinalRiskClassifier().to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        _, scores, _, row_idx, thresh_idx = run_epoch(m, lb_loader, None, None, device, train=False)
        probs = pairs_to_matrix(scores, row_idx, thresh_idx, len(lb_df))
        all_probs.append(probs)
        del m
        torch.cuda.empty_cache()

    avg_probs = np.mean(all_probs, axis=0)  # (N, 3)
    np.save(os.path.join(output_dir, "leaderboard_probs.npy"), avg_probs)

    pred_ranks = [decode_ordinal(row >= 0.5) for row in avg_probs]
    pred_labels = [RISK_LEVELS[r].capitalize() for r in pred_ranks]

    submission = pd.DataFrame({"row_id": lb_df["row_id"], "risk_level": pred_labels})
    print("\nPrediction distribution:")
    print(submission["risk_level"].value_counts().to_string())

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_nli_ordinal.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved -> {out_path}")


if __name__ == "__main__":
    main()
