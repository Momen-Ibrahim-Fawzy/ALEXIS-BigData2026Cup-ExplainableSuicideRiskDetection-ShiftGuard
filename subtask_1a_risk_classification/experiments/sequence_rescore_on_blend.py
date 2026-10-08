"""
Stack test: does sequence-rescoring (backlog item #1, validated true-OOF WF1
0.8014 -> 0.8078 on exp09 alone) still add value on TOP of the exp09+exp14
model ensemble (validated true-OOF WF1 0.8065, honest nested-CV)? If these
two independent techniques capture non-overlapping signal, stacking them
should beat both individually.

Uses w=0.35 (exp09 share) — the mode across nested-CV folds in
ensemble_exp09_exp14_nested.py (3/5 folds picked this value) — as a single
fixed, pre-committed blend weight, then trains the identical SeqRescorer
architecture from sequence_rescore.py on top of those blended probabilities.

Usage:
    python sequence_rescore_on_blend.py
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
from trainer import RISK2ID, ID2RISK, BASE_DIR, RESULTS_DIR
from sequence_rescore import SeqRescorer, UserSeqDataset, collate, build_users

EXP09_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
EXP14_DIR = os.path.join(RESULTS_DIR, "exp14_mental_roberta_large", "sequence_data")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

BLEND_W = 0.35
N_FOLDS = 5
EPOCHS = 60
PATIENCE = 8
LR = 3e-4
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
SEED = 42


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    train_meta = pd.read_csv(os.path.join(EXP09_DIR, "train_meta.csv"))
    exp09_probs = softmax(np.load(os.path.join(EXP09_DIR, "train_oof_logits.npy")))
    exp14_probs = softmax(np.load(os.path.join(EXP14_DIR, "train_oof_logits.npy")))
    blend_probs = BLEND_W * exp09_probs + (1 - BLEND_W) * exp14_probs
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])

    base_wf1 = f1_score(true_labels, blend_probs.argmax(1), average="weighted", zero_division=0)
    print(f"Blend alone (w={BLEND_W}, true OOF)  WF1={base_wf1:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = train_meta["anon_user_id"].values
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)

    all_rescored = np.zeros_like(blend_probs)
    fold_models = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]

        tr_users = build_users(tr_meta, blend_probs, "risk_norm")
        va_users = build_users(va_meta, blend_probs, "risk_norm")

        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True,
                            collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False,
                            collate_fn=lambda b: collate(b, True))

        model = SeqRescorer().to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b, labels_b = pad_mask_b.to(device), labels_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b)
                loss = criterion(out.reshape(-1, 4), labels_b.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                    pad_mask_b = pad_mask_b.to(device)
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

        print(f"  Fold {fold}: best val WF1={best_wf1:.4f} (epoch stopped {epoch+1})")
        model.load_state_dict(best_state)
        fold_models.append(model)

        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    n = len(ridx)
                    all_rescored[ridx] = out[bi, :n]

    rescored_preds = all_rescored.argmax(axis=1)
    rescored_wf1 = f1_score(true_labels, rescored_preds, average="weighted", zero_division=0)
    rescored_mf1 = f1_score(true_labels, rescored_preds, average="macro", zero_division=0)
    print(f"\nBlend alone WF1:        {base_wf1:.4f}")
    print(f"Blend+rescored WF1:     {rescored_wf1:.4f}  (true held-out OOF across all 5 folds)")
    print(f"Blend+rescored MF1:     {rescored_mf1:.4f}")
    print(f"vs exp09 alone:         0.8014")
    print(f"vs exp09 seq-rescored:  0.8078")
    print(f"vs blend alone (nested-CV honest): 0.8065")

    result = {
        "blend_alone_wf1": float(base_wf1),
        "blend_rescored_wf1": float(rescored_wf1),
        "blend_rescored_mf1": float(rescored_mf1),
        "exp09_alone": 0.8014, "exp09_seqrescored": 0.8078, "blend_nested_cv": 0.8065,
    }
    out_dir = os.path.join(RESULTS_DIR, "ensemble_exp09_exp14")
    with open(os.path.join(out_dir, "stacked_seqrescore_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    best_known = max(0.8078, 0.8065)
    if rescored_wf1 <= best_known:
        print(f"\n*** Stacking did NOT beat the best already-validated Sub1a result "
              f"({best_known:.4f}). Do not use for submission. ***")
        return

    # ── Apply to leaderboard ──────────────────────────────────────────────
    exp09_lb = softmax(np.load(os.path.join(EXP09_DIR, "leaderboard_logits.npy")))
    exp14_lb = softmax(np.load(os.path.join(EXP14_DIR, "leaderboard_logits.npy")))
    lb_blend = BLEND_W * exp09_lb + (1 - BLEND_W) * exp14_lb
    lb_meta = pd.read_csv(os.path.join(EXP09_DIR, "leaderboard_meta.csv"))

    lb_users = build_users(lb_meta.reset_index(drop=True), lb_blend, label_col=None)
    lb_ld = DataLoader(UserSeqDataset(lb_users), batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate(b, False))

    lb_rescored = np.zeros_like(lb_blend)
    lb_counts = np.zeros(len(lb_blend))
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, _, row_idx_b in lb_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
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

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_exp09_exp14_blend_seqrescored.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
