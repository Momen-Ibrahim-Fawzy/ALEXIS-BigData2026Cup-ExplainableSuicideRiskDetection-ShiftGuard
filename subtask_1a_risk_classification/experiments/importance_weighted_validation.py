"""Retrospective test: would importance-weighted validation (using the
population-shift diagnostic's adversarial classifier as sample weights) have
correctly predicted the real regressions this project has already observed?

If reweighting the true-OOF validation set to look more like the leaderboard
population (per the measured AUC=0.656 shift, confirmed above the AUC=0.507
null baseline) flips exp42 and the flan-t5 ensemble from "looks better than
exp09" to "looks worse than exp09" -- matching their real -0.0114 and -0.0031
regressions -- that's real evidence this methodology is more predictive than
plain unweighted true-OOF has been. If it doesn't change the picture, that
tells us reweighting validation alone isn't sufficient.

Steps:
  1. Fit the adversarial classifier (train vs leaderboard, exp09 embeddings)
     -- same as population_shift_diagnostic.py's check 1.
  2. Convert each training example's P(leaderboard) into an importance
     weight: w_i = P_lb / (1 - P_lb), clipped to avoid instability from
     near-certain predictions.
  3. Recompute exp42's true-OOF predictions fresh (checkpoints still exist).
  4. Load exp09's and the flan-t5-ensemble's already-cached true-OOF probs.
  5. Compare UNWEIGHTED vs WEIGHTED true-OOF WF1 for all three, against the
     known real deltas.

Usage:
    python importance_weighted_validation.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, DATA_DIR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR as _  # noqa -- sanity import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # subtask_1a_risk_classification
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_adversarial_weights():
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))

    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit(Xn, y)

    train_emb_n = (train_emb - mu) / sigma
    p_lb = clf.predict_proba(train_emb_n)[:, 1]  # P(this training example looks like leaderboard)

    # Classic covariate-shift importance weight: P(leaderboard)/P(train), clipped for stability
    p_lb_clipped = np.clip(p_lb, 0.02, 0.98)
    weights = p_lb_clipped / (1 - p_lb_clipped)
    weights = weights / weights.mean()  # normalize to mean 1, keeps F1 scale comparable
    weights = np.clip(weights, 0.1, 10.0)  # avoid a few extreme examples dominating

    print(f"  Importance weights: min={weights.min():.3f} max={weights.max():.3f} "
          f"mean={weights.mean():.3f} std={weights.std():.3f}")
    return weights


@torch.no_grad()
def get_exp42_oof_probs():
    cache_path = os.path.join(RESULTS_DIR, "exp42_train_oof_probs.npy")
    if os.path.exists(cache_path):
        print("  (using cached exp42 true-OOF probs)")
        return np.load(cache_path)

    tokenizer_name = "microsoft/deberta-v3-large"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=5)
    groups = df["anon_user_id"].values
    exp_dir = os.path.join(RESULTS_DIR, "exp42_deberta_large_long_context_clean")

    all_probs = np.zeros((len(df), 4))
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt = torch.load(os.path.join(exp_dir, f"best_fold{fold}.pt"), map_location=DEVICE, weights_only=True)
        model = Classifier(tokenizer_name).to(DEVICE).float()
        model.load_state_dict(ckpt)
        model.eval()
        ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]], tokenizer, 768)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
        probs = []
        for batch in ld:
            ids = batch["input_ids"].to(DEVICE); mask = batch["attention_mask"].to(DEVICE)
            logits = model(ids, mask)
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        all_probs[va_i] = np.concatenate(probs, axis=0)
        del model; torch.cuda.empty_cache()
        print(f"    exp42 fold {fold} done")

    np.save(cache_path, all_probs)
    return all_probs


def main():
    print(f"{'='*70}\n  Fitting adversarial-validation importance weights\n{'='*70}")
    weights = fit_adversarial_weights()

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])

    print(f"\n{'='*70}\n  Loading / computing true-OOF probabilities for each candidate\n{'='*70}")
    exp09_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)

    print("  exp42 (recomputing fresh from surviving checkpoints)...")
    exp42_probs = get_exp42_oof_probs()

    flant5_probs = np.load(os.path.join(RESULTS_DIR, "exp_flant5_large_qa_sft", "train_oof_scored_probs.npy"))
    flant5_ensemble_probs = 0.90 * exp09_probs + 0.10 * flant5_probs  # w=0.10, matches the real submission

    candidates = {
        "exp09 (baseline)": exp09_probs,
        "exp42 (REAL: -0.0114)": exp42_probs,
        "flan-t5 ensemble w=0.10 (REAL: -0.0031)": flant5_ensemble_probs,
    }

    print(f"\n{'='*70}\n  UNWEIGHTED vs IMPORTANCE-WEIGHTED true-OOF WF1\n{'='*70}")
    results = {}
    for name, probs in candidates.items():
        preds = probs.argmax(axis=1)
        unweighted = f1_score(true_labels, preds, average="weighted", zero_division=0)
        weighted = f1_score(true_labels, preds, average="weighted", sample_weight=weights, zero_division=0)
        results[name] = (unweighted, weighted)
        print(f"  {name}")
        print(f"    unweighted true-OOF WF1: {unweighted:.4f}")
        print(f"    IMPORTANCE-WEIGHTED WF1: {weighted:.4f}")

    exp09_u, exp09_w = results["exp09 (baseline)"]
    print(f"\n{'='*70}\n  DELTAS vs exp09 -- does weighting correctly flip sign to match REAL results?\n{'='*70}")
    for name, (u, w) in results.items():
        if name == "exp09 (baseline)":
            continue
        du, dw = u - exp09_u, w - exp09_w
        print(f"  {name}")
        print(f"    unweighted delta: {du:+.4f}  (this is what we actually saw before submitting)")
        print(f"    weighted delta:   {dw:+.4f}  (would this have warned us?)")
        if du > 0 and dw < 0:
            print(f"    -> WEIGHTING FLIPPED THE SIGN: would have correctly predicted the real regression!")
        elif du > 0 and dw > 0 and dw < du:
            print(f"    -> weighting reduced the apparent gain (didn't flip sign, but less misleading)")
        elif du > 0 and dw >= du:
            print(f"    -> weighting did NOT help -- still looks equally or more promising")
        else:
            print(f"    -> (unweighted was already negative or flat)")


if __name__ == "__main__":
    main()
