"""Fixed version of importance_weighted_validation_v42_v43.py.

The original version compared raw (non-rescored) checkpoint reloads for
exp09/v42/v43 against the REAL v42/v43 outcomes, which were built through
sequence_rescore.py's trained per-user residual-correction transformer.
Discovered during a diagnostic: the raw reload of exp09 alone scored
WF1=0.9134 vs the trusted rescored 0.8014 (85% prediction agreement) --
proof the raw and rescored pipelines are not comparable.

This version trains sequence_rescore.py's SeqRescorer on each candidate's
own blended logits (log(probs), a valid inverse for a softmax-based rescorer
that only needs a numeric per-class representation) using the exact same
GroupKFold splits, then compares TRUE-OOF rescored WF1 -- unweighted and
importance-weighted -- against exp09's own already-rescored, already-trusted
baseline (train_oof_logits.npy). This matches how the real submissions were
actually built and scored.

Usage:
    python importance_weighted_validation_v42_v43_fixed.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, DATA_DIR, RESULTS_DIR
from sequence_rescore import (SeqRescorer, UserSeqDataset, collate, build_users,
                               CLASS_WEIGHTS, EPOCHS, PATIENCE, LR, SEED, N_FOLDS)

SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V42_MEMBERS = ["exp09", "exp32_fgm", "exp33_selftrain", "exp36_freelb", "exp38_coral", "exp39_rdrop"]
V43_MEMBERS = ["exp09", "exp33_selftrain", "exp38_coral"]


def fit_adversarial_weights():
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))
    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit((X - mu) / sigma, y)
    p_lb = np.clip(clf.predict_proba((train_emb - mu) / sigma)[:, 1], 0.02, 0.98)
    w = p_lb / (1 - p_lb)
    w = w / w.mean()
    return np.clip(w, 0.1, 10.0)


def train_and_get_rescored_oof(blended_logits, train_meta, true_labels):
    """Train SeqRescorer via GroupKFold on blended_logits, return true-OOF
    rescored logits (same convention as sequence_rescore.py's main())."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    all_rescored = np.zeros_like(blended_logits)

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]

        tr_users = build_users(tr_meta, blended_logits, "risk_norm")
        va_users = build_users(va_meta, blended_logits, "risk_norm")

        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True,
                            collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False,
                            collate_fn=lambda b: collate(b, True))

        model = SeqRescorer().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(DEVICE), post_ids_b.to(DEVICE)
                pad_mask_b, labels_b = pad_mask_b.to(DEVICE), labels_b.to(DEVICE)
                out = model(logits_b, post_ids_b, pad_mask_b)
                loss = criterion(out.reshape(-1, 4), labels_b.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(DEVICE), post_ids_b.to(DEVICE)
                    pad_mask_b = pad_mask_b.to(DEVICE)
                    out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
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

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(DEVICE), post_ids_b.to(DEVICE)
                pad_mask_b = pad_mask_b.to(DEVICE)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    all_rescored[ridx] = out[bi, :n]
        print(f"    fold {fold}: val WF1={best_wf1:.4f}")

    return all_rescored


def main():
    print(f"{'='*70}\n  Fitting adversarial-validation importance weights\n{'='*70}")
    weights = fit_adversarial_weights()

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    train_meta = pd.read_csv(os.path.join(SEQ_DIR, "train_meta.csv"))
    assert (train_meta["anon_user_id"].values == df["anon_user_id"].values).all()

    print(f"\n{'='*70}\n  Loading cached raw member probs, building blends\n{'='*70}")
    member_probs = {}
    for key in sorted(set(V42_MEMBERS) | set(V43_MEMBERS)):
        p = np.load(os.path.join(RESULTS_DIR, f"{key}_train_oof_probs.npy"))
        member_probs[key] = np.clip(p, 1e-8, 1.0)
        print(f"  loaded {key}")

    v42_probs = np.mean([member_probs[k] for k in V42_MEMBERS], axis=0)
    v43_probs = np.mean([member_probs[k] for k in V43_MEMBERS], axis=0)
    v42_pseudologits = np.log(v42_probs)
    v43_pseudologits = np.log(v43_probs)

    exp09_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)
    eu = f1_score(true_labels, exp09_probs.argmax(1), average="weighted", zero_division=0)
    ew = f1_score(true_labels, exp09_probs.argmax(1), average="weighted", sample_weight=weights, zero_division=0)
    print(f"\n  exp09 (trusted, already rescored) unweighted={eu:.4f} weighted={ew:.4f}")

    results = {}
    for name, pseudologits in [("v42 6-way", v42_pseudologits), ("v43 3-way", v43_pseudologits)]:
        print(f"\n{'='*70}\n  Training SeqRescorer on {name} blend\n{'='*70}")
        rescored = train_and_get_rescored_oof(pseudologits, train_meta, true_labels)
        preds = rescored.argmax(axis=1)
        u = f1_score(true_labels, preds, average="weighted", zero_division=0)
        w = f1_score(true_labels, preds, average="weighted", sample_weight=weights, zero_division=0)
        results[name] = (u, w)
        print(f"  {name} rescored: unweighted={u:.4f} weighted={w:.4f}")

    print(f"\n{'='*70}\n  DELTAS vs exp09 (both sides now sequence-rescored)\n{'='*70}")
    for name, (u, w) in results.items():
        du, dw = u - eu, w - ew
        print(f"  {name}")
        print(f"    unweighted delta: {du:+.4f}")
        print(f"    weighted delta:   {dw:+.4f}")
        if du > 0 and dw < 0:
            print("    -> WEIGHTING FLIPPED THE SIGN")
        elif du > 0 and dw > 0 and dw < du:
            print("    -> weighting reduced the apparent gain")
        elif du > 0 and dw >= du:
            print("    -> weighting did NOT help")
        else:
            print("    -> unweighted was already negative/flat")

    real = {"v42 6-way": -0.0107, "v43 3-way": -0.0079}
    print(f"\n  (real outcomes for reference: v42={real['v42 6-way']:+.4f}, v43={real['v43 3-way']:+.4f})")


if __name__ == "__main__":
    main()
