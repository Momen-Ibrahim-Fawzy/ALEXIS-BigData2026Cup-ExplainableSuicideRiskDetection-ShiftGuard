"""
Dual-head CORAL + CE seq-rescorer for Sub1a (research-motivated extension).

Neither lineage combined CORAL (ordinal) loss with the sequence-rescoring
layer -- CORAL was only ever tried on the BASE exp09 classifier in isolation
(exp25, true OOF WF1 0.8018 vs baseline 0.8014, noise-level +0.0004). This
follows "Hierarchical Dual-Head Model for Suicide Risk Assessment via
MentalRoBERTa" (arXiv 2510.20085): a dual-head design (CORAL head + standard
classification head) trained jointly on a sequence-aware representation --
their ablations specifically attribute gains to the *combination* of
temporal modeling + dual-head + ordinal loss together, not any one part
alone. Our existing seq-rescorer already has the temporal (Transformer)
component; this adds the missing dual-head + CORAL piece on top of it,
rather than retrying CORAL on the (already input-frozen) base classifier.

CORAL (Cao et al. 2020): reformulate K-class ordinal problem as K-1 binary
"is rank > k" classifiers sharing a single weight vector, differing only in
bias -- guarantees monotonically consistent rank probabilities by
construction (unlike naive independent binary heads).

Gated the same way as every other experiment here: only trusted if true
combined-fold OOF WF1 beats the existing logits-only seq-rescorer's 0.8078.

Usage:
    python sequence_rescore_coral.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
N_FOLDS = 5
EPOCHS = 60
PATIENCE = 8
LR = 3e-4
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
SEED = 42
CORAL_WEIGHT = 0.5   # weight of CORAL loss relative to CE loss (tri-objective-style combo)
N_RANKS = 4          # indicator < ideation < behavior < attempt
N_THRESH = N_RANKS - 1


def coral_targets(labels):
    """labels: (N,) int in [0,3]. Returns (N, N_THRESH) binary: 1 if label > k."""
    thresholds = torch.arange(N_THRESH, device=labels.device).view(1, -1)
    return (labels.view(-1, 1) > thresholds).float()


def coral_loss(coral_logits, labels, ignore_mask):
    """coral_logits: (N, N_THRESH). labels: (N,). ignore_mask: (N,) True=keep."""
    targets = coral_targets(labels)
    loss = F.binary_cross_entropy_with_logits(coral_logits, targets, reduction="none").sum(-1)
    loss = loss[ignore_mask]
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=coral_logits.device)


def coral_probs_to_class_probs(coral_logits):
    """Convert K-1 cumulative sigmoid probs into K per-class probs.
    P(rank>k) monotonically decreasing by construction (approx, not enforced here);
    class prob_j = P(rank>j-1) - P(rank>j), with boundary P(rank>-1)=1, P(rank>K-1)=0."""
    cum_probs = torch.sigmoid(coral_logits)  # (N, K-1), P(rank > k)
    ones = torch.ones(cum_probs.shape[0], 1, device=cum_probs.device)
    zeros = torch.zeros(cum_probs.shape[0], 1, device=cum_probs.device)
    cum_probs_padded = torch.cat([ones, cum_probs, zeros], dim=1)  # (N, K+1)
    class_probs = cum_probs_padded[:, :-1] - cum_probs_padded[:, 1:]  # (N, K)
    return class_probs.clamp(min=0)


class SeqRescorerCoral(nn.Module):
    def __init__(self, n_classes=4, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dropout=DROPOUT, max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(n_classes, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)     # standard CE correction head
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)                # start as pure identity
        self.coral_head = nn.Linear(d_model, N_THRESH)    # ordinal head

    def forward(self, logits, post_ids, pad_mask):
        h = self.in_proj(logits) + self.pos_emb(post_ids.clamp(max=self.pos_emb.num_embeddings - 1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        ce_out = logits + self.out_proj(h)
        coral_out = self.coral_head(h)
        return ce_out, coral_out


class UserSeqDataset(Dataset):
    def __init__(self, users):
        self.users = users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        return self.users[i]


def collate(batch, has_labels=True):
    max_len = max(len(u["post_ids"]) for u in batch)
    B = len(batch)
    logits = torch.zeros(B, max_len, 4)
    post_ids = torch.zeros(B, max_len, dtype=torch.long)
    pad_mask = torch.ones(B, max_len, dtype=torch.bool)
    labels = torch.full((B, max_len), -100, dtype=torch.long) if has_labels else None
    row_idx = [None] * B
    for bi, u in enumerate(batch):
        n = len(u["post_ids"])
        logits[bi, :n] = torch.tensor(u["logits"], dtype=torch.float32)
        post_ids[bi, :n] = torch.tensor(u["post_ids"], dtype=torch.long)
        pad_mask[bi, :n] = False
        if has_labels:
            labels[bi, :n] = torch.tensor(u["labels"], dtype=torch.long)
        row_idx[bi] = u["row_indices"]
    return logits, post_ids, pad_mask, labels, row_idx


def build_users(meta_df, logits, label_col=None):
    users = []
    for uid, g in meta_df.groupby("anon_user_id"):
        g = g.sort_values("post_id")
        idx = g.index.to_numpy()
        u = {"logits": logits[idx], "post_ids": g["post_id"].to_numpy(), "row_indices": idx}
        if label_col:
            u["labels"] = np.array([RISK2ID[r] for r in g[label_col]])
        users.append(u)
    return users


def main():
    train_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    train_meta = pd.read_csv(os.path.join(SEQ_DIR, "train_meta.csv"))
    lb_logits = np.load(os.path.join(SEQ_DIR, "leaderboard_logits.npy"))
    lb_meta = pd.read_csv(os.path.join(SEQ_DIR, "leaderboard_meta.csv"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_preds = train_logits.argmax(axis=1)
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])
    base_wf1 = f1_score(true_labels, base_preds, average="weighted", zero_division=0)
    print(f"Baseline (exp09 alone, true OOF)  WF1={base_wf1:.4f}")
    print(f"Existing logits-only seq-rescorer WF1=0.8078 (baseline to beat)")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    all_rescored = np.zeros_like(train_logits)
    all_coral_probs = np.zeros((len(train_meta), 4))

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]

        tr_users = build_users(tr_meta, train_logits, "risk_norm")
        va_users = build_users(va_meta, train_logits, "risk_norm")

        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True,
                            collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False,
                            collate_fn=lambda b: collate(b, True))

        model = SeqRescorerCoral().to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        ce_criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b, labels_b = pad_mask_b.to(device), labels_b.to(device)
                ce_out, coral_out = model(logits_b, post_ids_b, pad_mask_b)

                flat_labels = labels_b.reshape(-1)
                keep = flat_labels != -100
                ce_loss = ce_criterion(ce_out.reshape(-1, 4), flat_labels)
                c_loss = coral_loss(coral_out.reshape(-1, N_THRESH), flat_labels.clamp(min=0), keep)
                loss = ce_loss + CORAL_WEIGHT * c_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                    pad_mask_b = pad_mask_b.to(device)
                    ce_out, coral_out = model(logits_b, post_ids_b, pad_mask_b)
                    ce_probs = F.softmax(ce_out, dim=-1)
                    coral_probs = coral_probs_to_class_probs(coral_out.reshape(-1, N_THRESH))
                    coral_probs = coral_probs.reshape(ce_out.shape)
                    combined = 0.5 * ce_probs + 0.5 * coral_probs
                    out = combined.cpu().numpy()
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

        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                ce_out, coral_out = model(logits_b, post_ids_b, pad_mask_b)
                ce_probs = F.softmax(ce_out, dim=-1)
                coral_probs = coral_probs_to_class_probs(coral_out.reshape(-1, N_THRESH)).reshape(ce_out.shape)
                combined = (0.5 * ce_probs + 0.5 * coral_probs).cpu().numpy()
                mask_np = ~pad_mask_b.cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    all_rescored[ridx] = combined[bi, :n]

    rescored_preds = all_rescored.argmax(axis=1)
    rescored_wf1 = f1_score(true_labels, rescored_preds, average="weighted", zero_division=0)
    rescored_mf1 = f1_score(true_labels, rescored_preds, average="macro", zero_division=0)
    print(f"\nBaseline (exp09 alone) WF1:        {base_wf1:.4f}")
    print(f"Existing logits-only rescorer WF1: 0.8078")
    print(f"CORAL dual-head rescorer WF1:      {rescored_wf1:.4f}  (true held-out OOF)")
    print(f"CORAL dual-head rescorer MF1:      {rescored_mf1:.4f}")
    print(f"Improvement vs existing rescorer:  {rescored_wf1 - 0.8078:+.4f} WF1")

    result = {"base_wf1": float(base_wf1), "existing_rescorer_wf1": 0.8078,
              "coral_dualhead_wf1": float(rescored_wf1), "coral_dualhead_mf1": float(rescored_mf1)}
    with open("sequence_rescore_coral_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved -> sequence_rescore_coral_result.json")


if __name__ == "__main__":
    main()
