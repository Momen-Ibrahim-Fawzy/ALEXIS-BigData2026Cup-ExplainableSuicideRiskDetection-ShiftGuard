"""
Generate leaderboard factor-probability predictions from the trained
dual-encoder Sub2 model (v14), in the same oof_probs.npy/leaderboard_probs.npy
format the other Sub2 sources use — so it can be dropped straight into the
existing ensemble scripts as a 4th source.

Usage:
    python predict_dual_encoder.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import load_leaderboard, get_tokenizer, FactorDataset
from dual_encoder_model import DualEncoderFactorModel
from factor_descriptions import FACTOR_DESCRIPTIONS
from train_dual_encoder import OUTPUT_DIR_DE, EMBED_DIM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    factor_enc = tokenizer(FACTOR_DESCRIPTIONS, max_length=64, padding="max_length",
                            truncation=True, return_tensors="pt")
    factor_ids = factor_enc["input_ids"].to(device)
    factor_mask = factor_enc["attention_mask"].to(device)

    lb_df = load_leaderboard()
    ds = FactorDataset(lb_df, tokenizer, MAX_LENGTH, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model_paths = [os.path.join(OUTPUT_DIR_DE, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS)
                   if os.path.exists(os.path.join(OUTPUT_DIR_DE, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} dual-encoder fold models")

    all_probs = []
    for p in model_paths:
        m = DualEncoderFactorModel(MODEL_NAME, embed_dim=EMBED_DIM).to(device).float()
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        fold_p = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                scores = m(ids, mask, factor_ids, factor_mask)
                fold_p.append(torch.sigmoid(scores.float()).cpu().numpy())
        all_probs.append(np.concatenate(fold_p))
        del m
        torch.cuda.empty_cache()

    avg = np.mean(all_probs, axis=0)
    np.save(os.path.join(OUTPUT_DIR_DE, "leaderboard_probs.npy"), avg)
    print(f"Saved leaderboard_probs.npy -> {OUTPUT_DIR_DE}  shape={avg.shape}")


if __name__ == "__main__":
    main()
