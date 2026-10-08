"""Controlled covariate-shift benchmark within RSD-15K, for the BigData 2026
paper's methodology section: does the same diagnostic pipeline validated on
the real PFA/leaderboard case (adversarial-validation AUC -> importance-
weighted CV) also work on a second, independently-constructed shift, where
we know the true "target" answer because RSD-15K is fully labeled?

Split: RSD-15K's 1,265 users split by median post length (word count) into
"short-form" (source, 639 users, 10,838 posts) and "long-form" (target, 626
users, 3,775 posts) -- a genuine covariate shift in writing style, which also
carries a correlated class-distribution shift (Indicator 35.4%->20.6%,
Attempt 4.5%->8.5%), reported honestly rather than claimed as a "pure"
isolated shift type: real shifts rarely separate cleanly.

Design, deliberately simpler than the main paper's methodology given time
budget: a single 90/10 (by user) train/validation split on source (not full
5-fold CV) for speed, DeBERTa-v3-base (not large) for speed. This is an
illustrative single controlled check, not a comprehensive shift-type
taxonomy -- described as such in the paper.

Usage:
    python rsd15k_controlled_shift.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
MODEL_NAME = "microsoft/deberta-v3-large"  # -base isn't fully downloaded on this machine
                                            # (large-file transfers stall regardless of size,
                                            # see paper Limitations); -large is already complete
                                            # and used throughout this project, so we reuse it here
                                            # for this side-experiment too rather than risk a stall.
OUT_DIR = os.path.join(RESULTS_DIR, "rsd15k_controlled_shift")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 384
SEED = 42
LR = 2e-5
EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 16
LABEL_SMOOTHING = 0.05


def compute_class_weights(labels, n_classes=4):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    inv_freq = counts.sum() / (n_classes * counts)
    return inv_freq / inv_freq.mean()


@torch.no_grad()
def predict_probs(model, texts, labels, tokenizer, batch_size=32):
    ds = TextDataset(texts, labels, tokenizer, MAX_LENGTH)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    all_probs = []
    model.eval()
    for batch in ld:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
            probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def run_epoch_train(model, loader, optimizer, scheduler, criterion):
    model.train()
    total = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
            loss = criterion(logits.float(), labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    rsd["n_words"] = rsd["text"].astype(str).str.split().apply(len)
    user_len = rsd.groupby("users")["n_words"].median()
    thresh = user_len.median()
    short_users = set(user_len[user_len <= thresh].index)
    long_users = set(user_len[user_len > thresh].index)

    source_df = rsd[rsd["users"].isin(short_users)].reset_index(drop=True)
    target_df = rsd[rsd["users"].isin(long_users)].reset_index(drop=True)
    print(f"source (short-form): {len(source_df)} posts, {source_df['users'].nunique()} users")
    print(f"target (long-form):  {len(target_df)} posts, {target_df['users'].nunique()} users")
    print("source class dist:\n", source_df["risk_norm"].value_counts(normalize=True))
    print("target class dist:\n", target_df["risk_norm"].value_counts(normalize=True))

    src_labels = np.array([RISK2ID[r] for r in source_df["risk_norm"]])
    tgt_labels = np.array([RISK2ID[r] for r in target_df["risk_norm"]])

    # ---- adversarial validation (TF-IDF, no GPU needed) ----
    print(f"\n{'='*70}\n  Adversarial validation: source vs target\n{'='*70}")
    all_text = pd.concat([source_df["text"], target_df["text"]]).astype(str)
    y_dom = np.concatenate([np.zeros(len(source_df)), np.ones(len(target_df))])
    tfidf = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3, ngram_range=(1, 2))
    X = tfidf.fit_transform(all_text)
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit(X, y_dom)
    p_dom = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_dom, p_dom)
    print(f"  adversarial AUC (source vs target): {auc:.4f}")

    p_target_given_source = np.clip(p_dom[:len(source_df)], 0.02, 0.98)
    weights = p_target_given_source / (1 - p_target_given_source)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, 10.0)
    print(f"  importance weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

    # ---- train on source (90/10 by user split), evaluate on source val + target ----
    print(f"\n{'='*70}\n  Training DeBERTa-v3-base on source\n{'='*70}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_i, va_i = next(gss.split(source_df, groups=source_df["users"].values))
    tr_df, va_df = source_df.iloc[tr_i], source_df.iloc[va_i]
    class_weights = compute_class_weights(src_labels[tr_i])
    print(f"  class weights: {class_weights.round(3)}")

    tr_ds = TextDataset(tr_df["text"].astype(str).tolist(), src_labels[tr_i].tolist(), tokenizer, MAX_LENGTH)
    va_ds = TextDataset(va_df["text"].astype(str).tolist(), src_labels[va_i].tolist(), tokenizer, MAX_LENGTH)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    torch.manual_seed(SEED)
    model = Classifier(MODEL_NAME).to(DEVICE).float()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(tr_ld) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
                                     label_smoothing=LABEL_SMOOTHING)

    best_wf1, best_state, patience_ctr = 0, None, 0
    for epoch in range(EPOCHS):
        train_loss = run_epoch_train(model, tr_ld, optimizer, scheduler, criterion)
        probs = predict_probs(model, va_df["text"].astype(str).tolist(), src_labels[va_i].tolist(), tokenizer)
        val_wf1 = f1_score(src_labels[va_i], probs.argmax(1), average="weighted", zero_division=0)
        print(f"  epoch {epoch+1}: train_loss={train_loss:.4f} val_wf1={val_wf1:.4f}", flush=True)
        if val_wf1 > best_wf1:
            best_wf1, patience_ctr = val_wf1, 0
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, "best.pt"))

    # source validation probs (for weighting) and target probs (real answer)
    src_val_probs = predict_probs(model, va_df["text"].astype(str).tolist(), src_labels[va_i].tolist(), tokenizer)
    tgt_probs = predict_probs(model, target_df["text"].astype(str).tolist(), tgt_labels.tolist(), tokenizer)

    src_val_wf1_unweighted = f1_score(src_labels[va_i], src_val_probs.argmax(1), average="weighted", zero_division=0)
    val_weights = weights[va_i]
    src_val_wf1_weighted = f1_score(src_labels[va_i], src_val_probs.argmax(1), average="weighted",
                                     sample_weight=val_weights, zero_division=0)
    tgt_wf1 = f1_score(tgt_labels, tgt_probs.argmax(1), average="weighted", zero_division=0)

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print(f"  source val WF1 (unweighted):        {src_val_wf1_unweighted:.4f}")
    print(f"  source val WF1 (importance-weighted): {src_val_wf1_weighted:.4f}")
    print(f"  target WF1 (real, held-out population): {tgt_wf1:.4f}")
    print(f"  adversarial AUC: {auc:.4f}")
    print(f"  gap (unweighted - target): {src_val_wf1_unweighted - tgt_wf1:+.4f}")
    print(f"  gap (weighted - target):   {src_val_wf1_weighted - tgt_wf1:+.4f}")

    result = {
        "source_val_wf1_unweighted": float(src_val_wf1_unweighted),
        "source_val_wf1_weighted": float(src_val_wf1_weighted),
        "target_wf1": float(tgt_wf1),
        "adversarial_auc": float(auc),
        "source_n": len(source_df), "target_n": len(target_df),
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
