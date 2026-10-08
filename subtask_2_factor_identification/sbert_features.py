"""Precompute fixed (non-fine-tuned) SBert semantic-similarity features
between each post and each of the 24 factor descriptions.

Different mechanism from the earlier dual-encoder experiment (v14): that one
trained its own text encoder end-to-end from scratch on our 1,635 examples
and failed badly (standalone OOF Macro F1 0.3801 -- an embedding space isn't
learnable from that little data). This instead uses a SentenceTransformers
model already pretrained on hundreds of millions of sentence pairs for
semantic similarity -- no fine-tuning, no risk of overfitting the embedding
space itself. The only thing learned downstream is how to weight these 24
fixed similarity scores alongside DeBERTa's own representation.

Output: (N, 24) cosine similarity matrix, one row per post, saved once for
train and once for leaderboard -- purely a precompute step, not part of the
training loop.

Usage:
    python sbert_features.py
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import *
from factor_descriptions import FACTOR_DESCRIPTIONS

SBERT_MODEL = "all-mpnet-base-v2"  # strong general-purpose STS model
OUT_DIR = os.path.join(os.path.dirname(__file__), "sbert_cache")


def compute_and_save(model, texts, out_path, factor_embeds):
    post_embeds = model.encode(texts, batch_size=64, show_progress_bar=True,
                                normalize_embeddings=True, convert_to_numpy=True)
    sims = post_embeds @ factor_embeds.T  # cosine sim, both sides L2-normalized -> (N, 24)
    np.save(out_path, sims.astype(np.float32))
    print(f"  Saved {sims.shape} -> {out_path}")
    return sims


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer(SBERT_MODEL)

    factor_embeds = model.encode(FACTOR_DESCRIPTIONS, normalize_embeddings=True, convert_to_numpy=True)
    np.save(os.path.join(OUT_DIR, "factor_embeds.npy"), factor_embeds)
    print(f"  Factor description embeddings: {factor_embeds.shape}")

    train_df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    compute_and_save(model, train_df["post"].astype(str).tolist(),
                      os.path.join(OUT_DIR, "train_sims.npy"), factor_embeds)
    train_df[["row_id"]].to_csv(os.path.join(OUT_DIR, "train_row_ids.csv"), index=False)

    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    compute_and_save(model, lb_df["post"].astype(str).tolist(),
                      os.path.join(OUT_DIR, "leaderboard_sims.npy"), factor_embeds)
    lb_df[["row_id"]].to_csv(os.path.join(OUT_DIR, "leaderboard_row_ids.csv"), index=False)

    print("\nDone. Note: oversample-source synthetic posts are NOT covered here "
          "(they'd need their own row_id-aligned sims) -- dataset_v2.py falls back "
          "to zero-similarity for any post_id not found in these caches, which in "
          "practice only affects synthetic training rows, never real train/leaderboard rows.")


if __name__ == "__main__":
    main()
