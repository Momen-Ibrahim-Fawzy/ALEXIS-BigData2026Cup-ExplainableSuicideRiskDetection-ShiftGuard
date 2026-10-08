"""Dedicated population-shift diagnostic for Subtask 1b, using 1b's OWN
representation space -- not borrowed from Sub1a's classification embeddings
(which were the only check ever applied to a 1b candidate, and failed to
flag either real Sub1b regression: the 4-way composition and denoised-solo).

Extracts mean-pooled backbone embeddings from the 'denoised' evidence-
extraction model (fold_0, a representative single fold -- this is a
diagnostic of representation geometry, not a leakage-sensitive prediction
task, so a single fold is a reasonable simplification) for every train.xlsx
and leaderboard.xlsx post, then runs the same adversarial-validation check
used for Sub1a: can a classifier tell train and leaderboard posts apart in
THIS model's own representation space?

Usage:
    python population_shift_diagnostic_1b.py
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import *
from dataset import get_tokenizer
from model import EvidenceExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAM_DIR = "outputs_denoised"
FOLD = 0


@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch_texts, max_length=MAX_LENGTH, padding=True, truncation=True,
                        return_tensors="pt").to(DEVICE)
        out = model.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    tokenizer = get_tokenizer()
    train_df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    print(f"train: {len(train_df)} posts, leaderboard: {len(lb_df)} posts")

    ckpt = torch.load(f"{FAM_DIR}/fold_{FOLD}/best.pt", weights_only=False, map_location=DEVICE)
    model = EvidenceExtractor().to(DEVICE).float()
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_emb = extract_embeddings(model, tokenizer, train_df["post"].tolist())
    lb_emb = extract_embeddings(model, tokenizer, lb_df["post"].tolist())
    print(f"embeddings extracted: train {train_emb.shape}, leaderboard {lb_emb.shape}")

    np.save("train_1b_embeddings_denoised_fold0.npy", train_emb)
    np.save("leaderboard_1b_embeddings_denoised_fold0.npy", lb_emb)

    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    clf = LogisticRegression(max_iter=2000, C=0.1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"\n{'='*70}\n  Sub1b adversarial validation (denoised model's own embedding space)\n{'='*70}")
    print(f"  5-fold CV AUC (train vs leaderboard): {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in aucs]}")
    print(f"  (for comparison: Sub1a's own embedding-space AUC was 0.656, TF-IDF AUC was 0.661)")

    # Also check: does the shift correlate with predicted evidence presence?
    # i.e. does the model predict "none" (no evidence) at different rates
    # for train vs leaderboard in a way that's suspicious?
    clf.fit(X, y)
    p_lb_train = clf.predict_proba(X[:len(train_emb)])[:, 1]
    print(f"\n  P(looks like leaderboard) on train posts: mean={p_lb_train.mean():.3f}, "
          f"std={p_lb_train.std():.3f}")


if __name__ == "__main__":
    main()
