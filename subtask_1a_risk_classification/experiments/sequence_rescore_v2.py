"""Embedding-enriched user-sequence re-scoring layer for Sub1a.

Extends the validated sequence_rescore.py (backlog item #1, the only lever
that's ever produced a confirmed real gain for Sub1a) rather than replacing
it: same residual-correction design, same GroupKFold discipline, same small
Transformer -- the only change is what each post is represented by going
into the sequence model. The original uses just the 4-dim softmax logit
vector per post; this concatenates the 1024-dim pooled DeBERTa embedding
(extract_embeddings.py) alongside it, so the trajectory model has access to
much richer per-post signal than a 4-way probability summary.

Motivated directly by the 2026-07-25 confusion-matrix diagnostic: several
of exp09's wrong predictions were short, decontextualized posts (e.g. a
post that is JUST a crisis-hotline phone number) where the 4-dim logit
vector alone genuinely doesn't carry enough information for the sequence
model to help -- but the richer embedding might, since it retains nuance
the final classification head discarded.

Deliberately kept as an independent file (not a flag on sequence_rescore.py)
so the original, already-confirmed-real, production-deployed script is
never at risk of being broken by this experiment.

Usage:
    python extract_logits.py         # if not already run
    python extract_embeddings.py     # if not already run
    python sequence_rescore_v2.py    # trains + evaluates + writes leaderboard CSV (only if it wins)
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, ID2RISK, RISK_LEVELS, BASE_DIR, RESULTS_DIR

EXP_NAME = "exp09_deberta_large_lr_high"
SEQ_DIR = os.path.join(RESULTS_DIR, EXP_NAME, "sequence_data")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
EMB_DROPOUT = 0.3   # extra regularization on the high-dim embedding input specifically --
                    # 153 users is a small sequence-training set for a ~1028-dim input
N_FOLDS = 5
EPOCHS = 60
PATIENCE = 8
LR = 3e-4
WEIGHT_DECAY = 3e-4  # slightly higher than v1's 1e-4 -- more input dims, more overfitting risk
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
SEED = 42

PRODUCTION_WF1 = 0.8078   # v1 sequence-rescoring, true OOF (currently deployed)
EXP09_BASELINE_WF1 = 0.8014


class SeqRescorerV2(nn.Module):
    def __init__(self, n_classes=4, emb_dim=1024, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT, emb_dropout=EMB_DROPOUT, max_len=64):
        super().__init__()
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.in_proj = nn.Linear(n_classes + emb_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  # start as pure identity (correction=0)

    def forward(self, logits, emb, post_ids, pad_mask):
        # logits: (B, L, 4)  emb: (B, L, emb_dim)  post_ids: (B, L)  pad_mask: (B, L) True=pad
        feat = torch.cat([logits, self.emb_dropout(emb)], dim=-1)
        h = self.in_proj(feat) + self.pos_emb(post_ids.clamp(max=self.pos_emb.num_embeddings - 1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        correction = self.out_proj(h)
        return logits + correction


class UserSeqDataset(Dataset):
    def __init__(self, users):
        self.users = users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        return self.users[i]


def collate(batch, emb_dim, has_labels=True):
    max_len = max(len(u["post_ids"]) for u in batch)
    B = len(batch)
    logits = torch.zeros(B, max_len, 4)
    emb = torch.zeros(B, max_len, emb_dim)
    post_ids = torch.zeros(B, max_len, dtype=torch.long)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    labels = torch.full((B, max_len), -100, dtype=torch.long) if has_labels else None
    row_idx = [None] * B
    for bi, u in enumerate(batch):
        n = len(u["post_ids"])
        logits[bi, :n] = torch.tensor(u["logits"], dtype=torch.float32)
        emb[bi, :n] = torch.tensor(u["emb"], dtype=torch.float32)
        post_ids[bi, :n] = torch.tensor(u["post_ids"], dtype=torch.long)
        pad_mask[bi, :n] = False
        if has_labels:
            labels[bi, :n] = torch.tensor(u["labels"], dtype=torch.long)
        row_idx[bi] = u["row_indices"]
    return logits, emb, post_ids, pad_mask, labels, row_idx


def build_users(meta_df, logits, emb, label_col=None):
    """meta_df must keep its ORIGINAL index (do not reset_index before calling) --
    same convention as v1's build_users, extended to also slice the embedding array."""
    users = []
    for uid, g in meta_df.groupby("anon_user_id"):
        g = g.sort_values("post_id")
        idx = g.index.to_numpy()
        u = {
            "logits": logits[idx],
            "emb": emb[idx],
            "post_ids": g["post_id"].to_numpy(),
            "row_indices": idx,
        }
        if label_col:
            u["labels"] = np.array([RISK2ID[r] for r in g[label_col]])
        users.append(u)
    return users


def main():
    train_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    train_meta = pd.read_csv(os.path.join(SEQ_DIR, "train_meta.csv"))
    lb_logits = np.load(os.path.join(SEQ_DIR, "leaderboard_logits.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))
    lb_meta = pd.read_csv(os.path.join(SEQ_DIR, "leaderboard_meta.csv"))
    emb_dim = train_emb.shape[1]
    print(f"Embedding dim: {emb_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_preds = train_logits.argmax(axis=1)
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])
    base_wf1 = f1_score(true_labels, base_preds, average="weighted", zero_division=0)
    print(f"Baseline (exp09 alone, true OOF)  WF1={base_wf1:.4f}")
    print(f"Compare to: production (v1 seq-rescore) {PRODUCTION_WF1:.4f}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    all_rescored = np.zeros_like(train_logits)
    fold_models = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]

        tr_users = build_users(tr_meta, train_logits, train_emb, "risk_norm")
        va_users = build_users(va_meta, train_logits, train_emb, "risk_norm")

        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True,
                            collate_fn=lambda b: collate(b, emb_dim, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False,
                            collate_fn=lambda b: collate(b, emb_dim, True))

        model = SeqRescorerV2(emb_dim=emb_dim).to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, emb_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, emb_b = logits_b.to(device), emb_b.to(device)
                post_ids_b, pad_mask_b = post_ids_b.to(device), pad_mask_b.to(device)
                labels_b = labels_b.to(device)
                out = model(logits_b, emb_b, post_ids_b, pad_mask_b)
                loss = criterion(out.reshape(-1, 4), labels_b.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, emb_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, emb_b = logits_b.to(device), emb_b.to(device)
                    post_ids_b, pad_mask_b = post_ids_b.to(device), pad_mask_b.to(device)
                    out = model(logits_b, emb_b, post_ids_b, pad_mask_b).cpu().numpy()
                    labels_np = labels_b.numpy()
                    mask_np = ~pad_mask_b.cpu().numpy()
                    preds_all.append(out[mask_np].argmax(axis=1))
                    labels_all.append(labels_np[mask_np])
            preds_all = np.concatenate(preds_all)
            labels_all = np.concatenate(labels_all)
            wf1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)

            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break

        print(f"  Fold {fold}: best val WF1={best_wf1:.4f} (epoch stopped {epoch+1})")
        model.load_state_dict(best_state)
        fold_models.append(model)

        model.eval()
        with torch.no_grad():
            for logits_b, emb_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, emb_b = logits_b.to(device), emb_b.to(device)
                post_ids_b, pad_mask_b = post_ids_b.to(device), pad_mask_b.to(device)
                out = model(logits_b, emb_b, post_ids_b, pad_mask_b).cpu().numpy()
                mask_np = ~pad_mask_b.cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    all_rescored[ridx] = out[bi, :n]

    rescored_preds = all_rescored.argmax(axis=1)
    rescored_wf1 = f1_score(true_labels, rescored_preds, average="weighted", zero_division=0)
    rescored_mf1 = f1_score(true_labels, rescored_preds, average="macro", zero_division=0)
    print(f"\nBaseline (exp09 alone) WF1:     {base_wf1:.4f}")
    print(f"Production (v1 rescore) WF1:    {PRODUCTION_WF1:.4f}")
    print(f"V2 (embedding-enriched) WF1:    {rescored_wf1:.4f}  (true held-out OOF across all 5 folds)")
    print(f"V2 MF1:                         {rescored_mf1:.4f}")
    print(f"Improvement vs exp09 alone:     {rescored_wf1 - base_wf1:+.4f}")
    print(f"Improvement vs production (v1): {rescored_wf1 - PRODUCTION_WF1:+.4f}")

    result = {
        "baseline_wf1": float(base_wf1),
        "production_v1_wf1": PRODUCTION_WF1,
        "v2_wf1": float(rescored_wf1),
        "v2_mf1": float(rescored_mf1),
        "improvement_vs_exp09": float(rescored_wf1 - base_wf1),
        "improvement_vs_production": float(rescored_wf1 - PRODUCTION_WF1),
    }
    with open(os.path.join(SEQ_DIR, "sequence_rescore_v2_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if rescored_wf1 <= PRODUCTION_WF1:
        print("\n*** V2 did NOT beat production (v1 seq-rescoring) on true OOF. "
              "Do not use it for a submission. ***")
        return

    print("\n*** V2 BEATS production -- generating leaderboard predictions. ***")
    lb_users = build_users(lb_meta.reset_index(drop=True), lb_logits, lb_emb, label_col=None)
    lb_ld = DataLoader(UserSeqDataset(lb_users), batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate(b, emb_dim, False))

    lb_rescored = np.zeros_like(lb_logits)
    lb_counts = np.zeros(len(lb_logits))
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            for logits_b, emb_b, post_ids_b, pad_mask_b, _, row_idx_b in lb_ld:
                logits_b, emb_b = logits_b.to(device), emb_b.to(device)
                post_ids_b, pad_mask_b = post_ids_b.to(device), pad_mask_b.to(device)
                out = model(logits_b, emb_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    lb_rescored[ridx] += out[bi, :n]
                    lb_counts[ridx] += 1
    lb_rescored /= lb_counts[:, None]

    pred_ids = lb_rescored.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    submission = pd.DataFrame({"row_id": lb_meta["row_id"], "risk_level": pred_labels})
    print("\nLeaderboard prediction distribution:")
    print(submission["risk_level"].value_counts().to_string())

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_exp09_deberta_large_lr_high_seqrescored_v2.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
