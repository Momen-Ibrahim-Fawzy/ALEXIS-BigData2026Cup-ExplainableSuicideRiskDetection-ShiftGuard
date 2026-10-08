"""
Generate leaderboard predictions from the trained context-aware Sub2 model
(v15), in the same oof_probs.npy/leaderboard_probs.npy format the other
Sub2 sources use.

Usage:
    python predict_context.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import *
from dataset_context import load_leaderboard_with_context, get_tokenizer, ContextFactorDataset
from model_context import FactorClassifierWithContext
from train_context import OUTPUT_DIR_CTX


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    df, context = load_leaderboard_with_context()
    ds = ContextFactorDataset(df["post"].values, context, None, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model_paths = [os.path.join(OUTPUT_DIR_CTX, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS)
                   if os.path.exists(os.path.join(OUTPUT_DIR_CTX, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} context-aware fold models")

    all_probs = []
    for p in model_paths:
        m = FactorClassifierWithContext(MODEL_NAME).to(device).float()
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        fold_p = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ctx = batch["context"].to(device)
                logits = m(ids, mask, ctx)
                fold_p.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_probs.append(np.concatenate(fold_p))
        del m
        torch.cuda.empty_cache()

    avg = np.mean(all_probs, axis=0)
    np.save(os.path.join(OUTPUT_DIR_CTX, "leaderboard_probs.npy"), avg)
    print(f"Saved leaderboard_probs.npy -> {OUTPUT_DIR_CTX}  shape={avg.shape}")


if __name__ == "__main__":
    main()
