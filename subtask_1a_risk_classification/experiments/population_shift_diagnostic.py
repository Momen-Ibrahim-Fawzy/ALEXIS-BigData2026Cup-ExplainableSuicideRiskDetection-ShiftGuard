"""Population-shift diagnostic: measure, rather than assume, how different
the leaderboard's 36 users are from the 153 training users.

Motivated by six real submissions in a row regressing this session, across
every technique category tried on both subtasks. The working hypothesis all
session has been "population shift" -- this is the first attempt to actually
test that hypothesis with evidence, instead of treating it as unfalsifiable.

Three independent checks:
  1. Adversarial validation: train a classifier to distinguish "is this post
     from train or leaderboard" using exp09's own cached true-OOF embeddings
     (the exact representation space the model actually reasons in -- no
     GPU needed, these are already computed). High AUC = real, measurable
     shift. Near-0.5 AUC = the two populations look similar in this space,
     which would point toward the metric-mismatch hypothesis instead.
  2. Surface-level distributional comparison: post length, vocabulary
     overlap (interpretable, unlike the embedding space).
  3. exp09's predicted class distribution on leaderboard vs. the TRUE class
     distribution on train -- not proof of shift by itself (could just be
     exp09 making systematic errors), but informative alongside 1 and 2.

Usage:
    python population_shift_diagnostic.py
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR
BASE_DIR = os.path.dirname(os.path.abspath(DATA_DIR))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")


def check_1_adversarial_validation_embeddings():
    print(f"\n{'='*70}\n  CHECK 1: Adversarial validation (exp09's own embedding space)\n{'='*70}")
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))
    print(f"  train embeddings: {train_emb.shape}, leaderboard embeddings: {lb_emb.shape}")

    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])

    # Standardize (logistic regression is scale-sensitive)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    clf = LogisticRegression(max_iter=2000, C=0.1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"  5-fold CV AUC (train vs leaderboard classifier): {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in aucs]}")
    print(f"  Interpretation: 0.5 = indistinguishable (no measurable shift in this space).")
    print(f"                  1.0 = perfectly separable (severe, systematic shift).")
    print(f"                  A well-known rule of thumb: AUC > 0.7 is considered a real,")
    print(f"                  actionable shift in the adversarial-validation literature.")
    return aucs.mean()


def check_2_length_and_vocab():
    print(f"\n{'='*70}\n  CHECK 2: Surface-level distributional comparison\n{'='*70}")
    train_df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")

    train_lens = train_df["post"].astype(str).str.split().apply(len)
    lb_lens = lb_df["post"].astype(str).str.split().apply(len)
    print(f"  Post length (words) -- train: mean={train_lens.mean():.1f} median={train_lens.median():.0f} "
          f"std={train_lens.std():.1f}")
    print(f"  Post length (words) -- leaderboard: mean={lb_lens.mean():.1f} median={lb_lens.median():.0f} "
          f"std={lb_lens.std():.1f}")
    from scipy import stats
    ks_stat, ks_p = stats.ks_2samp(train_lens, lb_lens)
    print(f"  KS test (length distributions differ?): statistic={ks_stat:.4f}, p={ks_p:.6f}")

    # Vocabulary overlap via TF-IDF top terms
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", min_df=3)
    all_text = pd.concat([train_df["post"].astype(str), lb_df["post"].astype(str)])
    tfidf.fit(all_text)
    train_vocab = set(tfidf.build_analyzer()(" ".join(train_df["post"].astype(str))))
    lb_vocab = set(tfidf.build_analyzer()(" ".join(lb_df["post"].astype(str))))
    overlap = len(train_vocab & lb_vocab) / max(len(train_vocab | lb_vocab), 1)
    print(f"  Vocabulary Jaccard overlap (top-2000 TF-IDF terms): {overlap:.4f}")

    return train_lens, lb_lens


def check_3_predicted_vs_true_class_distribution():
    print(f"\n{'='*70}\n  CHECK 3: exp09's predicted class distribution, leaderboard vs training truth\n{'='*70}")
    train_df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    train_df["risk_norm"] = train_df["suicide risk"].str.strip().str.lower()
    true_dist = train_df["risk_norm"].value_counts(normalize=True).sort_index()

    lb_logits = np.load(os.path.join(SEQ_DIR, "leaderboard_logits.npy"))
    RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]
    lb_preds = lb_logits.argmax(axis=1)
    pred_dist = pd.Series(lb_preds).map(lambda i: RISK_LEVELS[i]).value_counts(normalize=True).sort_index()

    print("  True distribution (train, 1635 posts):")
    print(true_dist.to_string())
    print("\n  exp09's predicted distribution (leaderboard, 378 posts):")
    print(pred_dist.to_string())

    combined = pd.DataFrame({"train_true": true_dist, "leaderboard_predicted": pred_dist}).fillna(0)
    max_diff = (combined["train_true"] - combined["leaderboard_predicted"]).abs().max()
    print(f"\n  Max class-proportion difference: {max_diff:.4f}")
    print("  (Not proof of shift by itself -- exp09 could simply be making systematic")
    print("   errors on leaderboard -- but informative alongside checks 1 and 2.)")


def check_4_adversarial_validation_tfidf():
    print(f"\n{'='*70}\n  CHECK 4: Adversarial validation (TF-IDF, interpretable cross-check)\n{'='*70}")
    train_df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")

    texts = pd.concat([train_df["post"].astype(str), lb_df["post"].astype(str)]).reset_index(drop=True)
    y = np.concatenate([np.zeros(len(train_df)), np.ones(len(lb_df))])

    tfidf = TfidfVectorizer(max_features=3000, stop_words="english", min_df=3, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"  5-fold CV AUC (TF-IDF-based): {aucs.mean():.4f} +/- {aucs.std():.4f}")

    # Fit on all data to inspect top distinguishing features
    clf.fit(X, y)
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    top_leaderboard = feature_names[np.argsort(coefs)[-15:][::-1]]
    top_train = feature_names[np.argsort(coefs)[:15]]
    print(f"\n  Top terms MORE common in leaderboard posts: {list(top_leaderboard)}")
    print(f"  Top terms MORE common in training posts: {list(top_train)}")

    return aucs.mean()


def main():
    auc_emb = check_1_adversarial_validation_embeddings()
    check_2_length_and_vocab()
    check_3_predicted_vs_true_class_distribution()
    auc_tfidf = check_4_adversarial_validation_tfidf()

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  Embedding-space adversarial AUC: {auc_emb:.4f}")
    print(f"  TF-IDF adversarial AUC:          {auc_tfidf:.4f}")
    if max(auc_emb, auc_tfidf) > 0.7:
        print("  -> Real, measurable population shift detected. This supports the")
        print("     'population shift' hypothesis for why offline validation hasn't")
        print("     transferred, and tells us it's detectable from text/embeddings alone.")
    elif max(auc_emb, auc_tfidf) < 0.6:
        print("  -> Train and leaderboard posts look statistically similar in both")
        print("     representation spaces tested. This weakens the population-shift")
        print("     explanation and makes the metric-implementation-mismatch hypothesis")
        print("     relatively more plausible as an explanation for the 6/6 failures.")
    else:
        print("  -> Mild, ambiguous signal -- some shift, but not dramatic. Inconclusive")
        print("     on its own; consider alongside checks 2 and 3 above.")


if __name__ == "__main__":
    main()
