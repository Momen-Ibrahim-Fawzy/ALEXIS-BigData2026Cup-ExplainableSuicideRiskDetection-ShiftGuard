"""Importance-weighted check for exp47 (RSD-15K two-stage transfer learning),
the strongest true-OOF Sub1a result of the whole session (0.8177 vs exp09's
0.8003, +0.0174). Given every prior candidate that looked good on raw
true-OOF has regressed for real (7/7) or come back flat (exp46), this is the
mandatory next gate before this is even considered for a submission.

Usage:
    python importance_weighted_validation_exp47.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, DATA_DIR, RESULTS_DIR

MODEL_NAME = "microsoft/deberta-v3-large"
EXP_DIR = os.path.join(RESULTS_DIR, "exp47_rsd15k_two_stage")
SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
MAX_LENGTH = 512
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@torch.no_grad()
def get_exp47_oof_probs(df, groups, gkf, tokenizer):
    cache_path = os.path.join(RESULTS_DIR, "exp47_train_oof_probs.npy")
    if os.path.exists(cache_path):
        print("  (using cached exp47 true-OOF probs)")
        return np.load(cache_path)

    all_probs = np.zeros((len(df), 4))
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        model = Classifier(MODEL_NAME).to(DEVICE).float()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]], tokenizer, MAX_LENGTH)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
        probs = []
        for batch in ld:
            ids = batch["input_ids"].to(DEVICE); mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
                p = torch.softmax(logits.float(), dim=-1)
            probs.append(p.cpu().numpy())
        all_probs[va_i] = np.concatenate(probs, axis=0)
        del model
        torch.cuda.empty_cache()
        print(f"    exp47 fold {fold} done", flush=True)

    np.save(cache_path, all_probs)
    return all_probs


def main():
    print(f"{'='*70}\n  Fitting adversarial-validation importance weights\n{'='*70}")
    weights = fit_adversarial_weights()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    groups = df["anon_user_id"].values
    gkf = GroupKFold(n_splits=5)

    print(f"\n{'='*70}\n  Computing exp47 true-OOF probs\n{'='*70}")
    exp47_probs = get_exp47_oof_probs(df, groups, gkf, tokenizer)

    exp09_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)

    print(f"\n{'='*70}\n  UNWEIGHTED vs IMPORTANCE-WEIGHTED true-OOF WF1\n{'='*70}")
    for name, probs in [("exp09 (baseline)", exp09_probs), ("exp47 (RSD-15K two-stage)", exp47_probs)]:
        preds = probs.argmax(axis=1)
        u = f1_score(true_labels, preds, average="weighted", zero_division=0)
        w = f1_score(true_labels, preds, average="weighted", sample_weight=weights, zero_division=0)
        print(f"  {name}: unweighted={u:.4f}  weighted={w:.4f}")
        if name == "exp09 (baseline)":
            eu, ew = u, w
        else:
            du, dw = u - eu, w - ew
            print(f"\n  DELTA vs exp09: unweighted={du:+.4f}  weighted={dw:+.4f}")
            if du > 0 and dw < 0:
                print("  -> WEIGHTING FLIPPED THE SIGN -- same warning pattern as exp42 and flan-t5-xl-solo, both real regressions")
            elif du > 0 and dw > 0 and dw < du:
                print("  -> weighting reduced the apparent gain but did not flip it -- ambiguous, same pattern as v42/v43/flan-t5-ensemble (all of which also regressed for real)")
            elif du > 0 and dw >= du:
                print("  -> weighting did NOT reduce the apparent gain at all -- the first candidate this session where reweighting doesn't undercut the raw signal")
            else:
                print("  -> unweighted was already negative/flat")


if __name__ == "__main__":
    main()
