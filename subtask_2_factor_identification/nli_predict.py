"""Full Stage-0 zero-shot pipeline: score train + leaderboard posts with the
off-the-shelf NLI entailment checkpoint (NO fine-tuning), tune per-factor
thresholds, and produce a submission-ready factor_predictions.csv.

This is a direct test of whether the zero-shot signal (validated via the
gap diagnostic in nli_entailment_check.py) actually holds up on the real
leaderboard -- the fastest way to find out, rather than iterating further
on OOF-style diagnostics we've already shown can be misleading (v18).

Threshold tuning uses the pseudo-leaderboard-honest split (same as
predict_v2.py) even though the model itself was never fit to our data at
all -- the only thing being tuned to our data here is a single per-factor
scalar threshold, much lower capacity than anything else tried this
session, but we keep the same discipline for consistency.

Usage:
    python nli_predict.py
"""

import argparse, ast, json, os, time
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import f1_score

from config import *
from factor_descriptions import FACTOR_DESCRIPTIONS_BY_NAME
from pseudo_leaderboard_split import get_or_create_split

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
HYPOTHESIS_TEMPLATE = "This example shows: {}"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_nli_zeroshot")


def score_posts(classifier, texts, candidate_descriptions, tag):
    scores = np.zeros((len(texts), len(candidate_descriptions)), dtype=np.float32)
    t0 = time.time()
    for i, text in enumerate(texts):
        result = classifier(text, candidate_descriptions, multi_label=True,
                             hypothesis_template=HYPOTHESIS_TEMPLATE)
        score_by_desc = dict(zip(result["labels"], result["scores"]))
        for j, desc in enumerate(candidate_descriptions):
            scores[i, j] = score_by_desc[desc]
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(texts) - i - 1) / rate
            print(f"  [{tag}] {i+1}/{len(texts)}  ({rate:.1f} posts/s, ETA {eta/60:.1f} min)")
    return scores


def tune_thresholds(probs, labels):
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    for i in range(NUM_FACTORS):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def compute_macro_f1(probs, labels, thresholds):
    preds = (probs >= thresholds).astype(int)
    return float(np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    candidate_descriptions = [FACTOR_DESCRIPTIONS_BY_NAME[f] for f in FACTOR_LIST]

    train_df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    train_df["factors_list"] = train_df["factors"].apply(
        lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])
    labels = np.zeros((len(train_df), NUM_FACTORS), dtype=np.float32)
    for i, row in train_df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                labels[i, FACTOR2ID[fl]] = 1.0

    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")

    train_scores_path = os.path.join(OUT_DIR, "train_scores.npy")
    lb_scores_path = os.path.join(OUT_DIR, "leaderboard_scores.npy")

    if os.path.exists(train_scores_path) and os.path.exists(lb_scores_path):
        print("Scores already cached, skipping inference.")
        train_scores = np.load(train_scores_path)
        lb_scores = np.load(lb_scores_path)
    else:
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading {NLI_MODEL} (device={'cuda' if device==0 else 'cpu'})...")
        classifier = pipeline("zero-shot-classification", model=NLI_MODEL, device=device)

        print(f"Scoring {len(train_df)} train posts...")
        train_scores = score_posts(classifier, train_df["post"].astype(str).tolist(), candidate_descriptions, "train")
        np.save(train_scores_path, train_scores)

        print(f"Scoring {len(lb_df)} leaderboard posts...")
        lb_scores = score_posts(classifier, lb_df["post"].astype(str).tolist(), candidate_descriptions, "leaderboard")
        np.save(lb_scores_path, lb_scores)

    np.save(os.path.join(OUT_DIR, "train_labels.npy"), labels)
    train_df[["row_id", "anon_user_id"]].to_csv(os.path.join(OUT_DIR, "train_meta.csv"), index=False)

    # Threshold tuning: pseudo-leaderboard-honest (see module docstring for why
    # this is lower-risk here than elsewhere, but kept for consistency)
    holdout_users = set(get_or_create_split())
    is_holdout = train_df["anon_user_id"].isin(holdout_users).values
    thresholds = tune_thresholds(train_scores[~is_holdout], labels[~is_holdout])
    honest_score = compute_macro_f1(train_scores[is_holdout], labels[is_holdout], thresholds)
    naive_thresholds = tune_thresholds(train_scores, labels)
    naive_score = compute_macro_f1(train_scores, labels, naive_thresholds)

    print(f"\nZero-shot NLI (no fine-tuning) results:")
    print(f"  Naive tuned Macro F1 (all 153 users):        {naive_score:.4f}")
    print(f"  Honest pseudo-leaderboard Macro F1 (20 held): {honest_score:.4f}")
    print(f"  (compare to factor_verified baseline: 0.5806, current best 2-way ensemble: 0.5982)")

    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump({"naive_score": float(naive_score), "honest_score": float(honest_score)}, f, indent=2)

    # Use the honest (pseudo-leaderboard-tuned) thresholds for the actual submission --
    # consistent with predict_v2.py's default.
    preds = (lb_scores >= thresholds).astype(int)
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"], "factors": str(factors) if factors else "[]"})
    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "factor_predictions_nli_zeroshot.csv")
    out.to_csv(out_path, index=False)
    n = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"\nSaved -> {out_path}  (mean factors/post: {n.mean():.2f}, zero: {(n==0).sum()}/{len(out)})")


if __name__ == "__main__":
    main()
