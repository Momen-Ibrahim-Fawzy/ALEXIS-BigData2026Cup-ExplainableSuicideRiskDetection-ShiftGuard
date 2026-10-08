"""Quick hyperparameter sweep for the sequence-rescorer's OWN architecture --
an axis neither session ever revisited once the original D_MODEL=64/2-layer/
4-head/patience=8 config was found to work. Reuses exp28's already-extracted
true-OOF logits (no DeBERTa retraining needed), so each config trains in
minutes, not hours.

Usage:
    python sweep_rescorer_hparams.py
"""

import itertools, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, RISK_LEVELS, RESULTS_DIR

EXP_NAME = "exp28_deberta_large_lrhigh_patience8"
SEQ_DIR = os.path.join(RESULTS_DIR, EXP_NAME, "sequence_data")

N_FOLDS = 5
LR = 3e-4
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
SEED = 42
BASELINE_WF1 = 0.8145  # exp28 + proven D_MODEL=64/2L/4H/patience=8 rescorer

CONFIGS = [
    {"name": "baseline (D64/2L/4H/pat8)",     "d_model": 64,  "n_layers": 2, "n_heads": 4, "patience": 8,  "epochs": 60},
    {"name": "3 layers",                       "d_model": 64,  "n_layers": 3, "n_heads": 4, "patience": 8,  "epochs": 60},
    {"name": "8 heads",                        "d_model": 64,  "n_layers": 2, "n_heads": 8, "patience": 8,  "epochs": 60},
    {"name": "patience=16, epochs=120",        "d_model": 64,  "n_layers": 2, "n_heads": 4, "patience": 16, "epochs": 120},
    {"name": "d_model=32 (more regularized)",  "d_model": 32,  "n_layers": 2, "n_heads": 4, "patience": 8,  "epochs": 60},
    {"name": "d_model=96",                     "d_model": 96,  "n_layers": 2, "n_heads": 4, "patience": 8,  "epochs": 60},
]


class SeqRescorer(nn.Module):
    def __init__(self, n_classes=4, d_model=64, n_heads=4, n_layers=2, dropout=0.1, max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(n_classes, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, logits, post_ids, pad_mask):
        h = self.in_proj(logits) + self.pos_emb(post_ids.clamp(max=self.pos_emb.num_embeddings - 1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return logits + self.out_proj(h)


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


def run_config(cfg, train_logits, train_meta, true_labels, device, class_w):
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values
    all_rescored = np.zeros_like(train_logits)

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]
        tr_users = build_users(tr_meta, train_logits, "risk_norm")
        va_users = build_users(va_meta, train_logits, "risk_norm")
        tr_ld = DataLoader(tr_users, batch_size=16, shuffle=True, collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(va_users, batch_size=32, shuffle=False, collate_fn=lambda b: collate(b, True))

        model = SeqRescorer(d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"]).to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(cfg["epochs"]):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b, labels_b = pad_mask_b.to(device), labels_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b)
                loss = criterion(out.reshape(-1, 4), labels_b.reshape(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                    pad_mask_b = pad_mask_b.to(device)
                    out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                    mask_np = ~pad_mask_b.cpu().numpy()
                    preds_all.append(out[mask_np].argmax(axis=1))
                    labels_all.append(labels_b.numpy()[mask_np])
            preds_all, labels_all = np.concatenate(preds_all), np.concatenate(labels_all)
            wf1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= cfg["patience"]:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    all_rescored[ridx] = out[bi, :len(ridx)]
        del model
        torch.cuda.empty_cache()

    preds = all_rescored.argmax(axis=1)
    return f1_score(true_labels, preds, average="weighted", zero_division=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    train_meta = pd.read_csv(os.path.join(SEQ_DIR, "train_meta.csv"))
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    print(f"Baseline (proven config): {BASELINE_WF1:.4f}\n")
    results = []
    for cfg in CONFIGS:
        wf1 = run_config(cfg, train_logits, train_meta, true_labels, device, class_w)
        print(f"  {cfg['name']:35s}  true-OOF WF1 = {wf1:.4f}  ({wf1-BASELINE_WF1:+.4f} vs baseline)")
        results.append({"name": cfg["name"], "wf1": wf1})

    results.sort(key=lambda r: -r["wf1"])
    print(f"\n{'='*60}\nRANKED\n{'='*60}")
    for r in results:
        print(f"  {r['wf1']:.4f}  {r['name']}")

    with open(os.path.join(SEQ_DIR, "rescorer_hparam_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
