"""Extract per-post POOLED EMBEDDINGS (not just the 4-dim logit summary) for
exp09, for both the true-OOF train set and the leaderboard set. Feeds
sequence_rescore_v2.py.

Why: sequence_rescore.py's SeqRescorer currently sees only a 4-dim softmax-
logit vector per post -- a heavily compressed summary that discards almost
everything DeBERTa computed about that post. A confusion-matrix diagnostic
(2026-07-25) found several wrong predictions caused by short, decontextual-
ized posts (e.g. a post that is JUST a crisis-hotline phone number,
predicted "attempt" with no other signal available) -- exactly the case
where richer per-post context, combined with the *sequence* mechanism that's
already proven to help, should matter most. This script extracts the
1024-dim mean-pooled hidden state (the same vector Classifier.forward()
computes right before its final Linear(1024,4) head) so the rescorer can use
it instead of/alongside the 4-dim logits.

Train embeddings are genuine out-of-fold (same discipline as extract_logits.py).

Usage:
    python extract_embeddings.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, BASE_DIR, DATA_DIR, RESULTS_DIR
from experiment_config import EXPERIMENTS

EXP_NAME = "exp09_deberta_large_lr_high"
N_FOLDS = 5
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME, "sequence_data")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def pooled_embedding(model, ids, mask):
    """Replicates Classifier.forward()'s pooling exactly, stopping one layer
    short of the classification head -- no architecture change, no retrain."""
    out = model.backbone(input_ids=ids, attention_mask=mask)
    h = out.last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EXPERIMENTS[EXP_NAME]
    exp_dir = os.path.join(RESULTS_DIR, EXP_NAME)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    hidden = None

    # ── Train: true OOF embeddings ──────────────────────────────────────────
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    oof_emb = None
    filled = np.zeros(len(df), dtype=bool)

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        print(f"  [train OOF] fold {fold}: {len(va_i)} rows <- {ckpt_path}")

        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        if hidden is None:
            hidden = model.backbone.config.hidden_size
            oof_emb = np.zeros((len(df), hidden), dtype=np.float32)

        va_ds = TextDataset(va_df["post"].tolist(),
                             [RISK2ID[r] for r in va_df["risk_norm"]],
                             tokenizer, cfg["max_length"])
        va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                            num_workers=2, pin_memory=True)

        emb_list = []
        for batch in va_ld:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb_list.append(pooled_embedding(model, ids, mask).float().cpu().numpy())
        oof_emb[va_i] = np.concatenate(emb_list, axis=0)
        filled[va_i] = True
        del model
        torch.cuda.empty_cache()

    assert filled.all(), "Not all train rows received OOF embeddings"

    np.save(os.path.join(OUT_DIR, "train_oof_embeddings.npy"), oof_emb)
    print(f"  Saved train OOF embeddings: {oof_emb.shape} -> {OUT_DIR}/train_oof_embeddings.npy")

    # ── Leaderboard: 5-fold ensemble average embedding ──────────────────────
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    lb_ds = TextDataset(lb_df["post"].tolist(), [0] * len(lb_df), tokenizer, cfg["max_length"])
    lb_ld = DataLoader(lb_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    fold_embs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        print(f"  [leaderboard] fold {fold} <- {ckpt_path}")
        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        emb_list = []
        for batch in lb_ld:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb_list.append(pooled_embedding(model, ids, mask).float().cpu().numpy())
        fold_embs.append(np.concatenate(emb_list, axis=0))
        del model
        torch.cuda.empty_cache()

    lb_emb = np.mean(fold_embs, axis=0)
    np.save(os.path.join(OUT_DIR, "leaderboard_embeddings.npy"), lb_emb)
    print(f"  Saved leaderboard embeddings: {lb_emb.shape} -> {OUT_DIR}/leaderboard_embeddings.npy")


if __name__ == "__main__":
    main()
