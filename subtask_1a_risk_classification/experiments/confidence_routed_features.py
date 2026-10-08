"""Confidence-based routing with engineered psychological features, for
Sub1a's most ambiguous predictions. Different mechanism from everything
else tried this session: not a different backbone, not more training data,
not a probability blend -- a rule-based override using interpretable,
clinically-motivated features (method specificity, temporal immediacy,
protective factors, access to means -- standard C-SSRS-adjacent concepts)
for the small minority of cases where exp09's own confidence margin
(top1 - top2 predicted probability) is low.

Motivation: exp09 is highly confident on ~94% of rows (median margin 0.87)
and only genuinely uncertain on ~5-6% -- exactly the kind of "implicit /
ambiguous" cases a recent paper on this same competition series flagged as
the hard case standard deep classifiers miss. Simple keyword/pattern
features may generalize more robustly across populations than the deep
model's learned representations specifically for these hard cases, since
they encode direct clinical signal rather than something learned from only
153 users' writing style.

Deliberately ONE principled design, no threshold/feature sweep: margin
threshold 0.2 (chosen because it's where the confidence distribution's mass
thins out, not tuned against this metric), one fixed feature set motivated
by C-SSRS-adjacent clinical concepts, one secondary model (multinomial
logistic regression, low-capacity by design to avoid overfitting the
already-small ambiguous subset).

Usage:
    python confidence_routed_features.py
"""

import os, re, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, DATA_DIR, RESULTS_DIR

SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
MARGIN_THRESHOLD = 0.2
CLASS_WEIGHTS = {0: 0.669, 1: 0.788, 2: 1.045, 3: 3.586}  # indicator, ideation, behavior, attempt

FEATURE_LEXICON = {
    "method_specific": [
        r"\boverdos\w*", r"\bpills?\b", r"\brope\b", r"\bhang\w*", r"\bgun\b", r"\bjump\w*",
        r"\bbridge\b", r"\bcut(ting)?\b", r"\bwrist\b", r"\bbleed\w*", r"\bpoison\w*",
        r"\bsuffocat\w*", r"\bpainkillers?\b", r"\brailway\b", r"\btrain track",
    ],
    "temporal_immediacy": [
        r"\btonight\b", r"\btoday\b", r"\btomorrow\b", r"\bplan(ning|ned)?\b",
        r"\bgoing to\b", r"\babout to\b", r"\bready to\b", r"\bright now\b", r"\bsoon\b",
    ],
    "past_attempt": [
        r"\btried\b", r"\battempt(ed|ing)?\b", r"\btook\b", r"\bswallowed\b",
        r"\bsurvived\b", r"\bwoke up\b", r"\bfailed\b",
    ],
    "direct_suicide_statement": [
        r"\bkill(ing)? myself\b", r"\bend(ing)? my life\b", r"\bsuicide\b",
        r"\bcommit(ting)? suicide\b", r"\bwant to die\b", r"\bdie by suicide\b",
    ],
    "protective_factors": [
        r"\bfamily\b", r"\bfriends?\b", r"\bhelp\b", r"\bhope\b", r"\btherap\w*",
        r"\bsupport\b", r"\blove\b", r"\bfuture\b", r"\breasons? to live\b", r"\bgetting better\b",
    ],
    "vague_distress": [
        r"\bsad\b", r"\bhopeless\b", r"\bworthless\b", r"\btired\b", r"\bpain\b",
        r"\bempty\b", r"\balone\b", r"\bno one\b", r"\bnobody\b",
    ],
    "access_to_means": [
        r"\bhave (a |some )?(pills|gun|rope|knife)\b", r"\bbought\b", r"\bown a\b",
        r"\baccess to\b", r"\bin my (hand|room|drawer)\b",
    ],
}


def extract_features(texts):
    rows = []
    for t in texts:
        t = str(t)
        tl = t.lower()
        words = tl.split()
        n_words = max(len(words), 1)
        feat = {}
        for cat, patterns in FEATURE_LEXICON.items():
            count = sum(len(re.findall(p, tl)) for p in patterns)
            feat[cat] = count / n_words * 100  # per-100-words rate
            feat[f"{cat}_present"] = float(count > 0)
        feat["n_words"] = n_words
        feat["n_exclaim"] = t.count("!")
        feat["n_question"] = t.count("?")
        feat["n_caps_words"] = sum(1 for w in t.split() if len(w) > 2 and w.isupper())
        rows.append(feat)
    df = pd.DataFrame(rows)
    return df


def main():
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    groups = df["anon_user_id"].values

    exp09_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)
    exp09_preds = exp09_probs.argmax(axis=1)
    sorted_probs = np.sort(exp09_probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    ambiguous = margin < MARGIN_THRESHOLD
    print(f"Ambiguous rows (margin < {MARGIN_THRESHOLD}): {ambiguous.sum()}/{len(df)} "
          f"({ambiguous.mean()*100:.1f}%)")

    feat_df = extract_features(df["post"].tolist())
    print(f"Engineered features: {feat_df.shape[1]} columns")

    exp09_wf1 = f1_score(labels, exp09_preds, average="weighted", zero_division=0)
    print(f"\nexp09 baseline true-OOF WF1: {exp09_wf1:.4f}")

    gkf = GroupKFold(n_splits=5)
    routed_preds = exp09_preds.copy()
    feat_only_preds = np.zeros(len(df), dtype=int)

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat_df.iloc[tr_i].values)
        X_va = scaler.transform(feat_df.iloc[va_i].values)

        sample_w = np.array([CLASS_WEIGHTS[y] for y in labels[tr_i]])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(X_tr, labels[tr_i], sample_weight=sample_w)

        va_feat_preds = clf.predict(X_va)
        feat_only_preds[va_i] = va_feat_preds

        va_ambiguous = ambiguous[va_i]
        va_indices = np.array(va_i)
        override_indices = va_indices[va_ambiguous]
        routed_preds[override_indices] = va_feat_preds[va_ambiguous]
        print(f"  fold {fold}: {va_ambiguous.sum()} ambiguous rows overridden "
              f"(of {len(va_i)} val rows)")

    feat_only_wf1 = f1_score(labels, feat_only_preds, average="weighted", zero_division=0)
    routed_wf1 = f1_score(labels, routed_preds, average="weighted", zero_division=0)

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print(f"  exp09 alone (baseline):              WF1={exp09_wf1:.4f}")
    print(f"  feature-only model (all rows, ref):   WF1={feat_only_wf1:.4f}")
    print(f"  ROUTED (exp09 + feature override on ambiguous rows): WF1={routed_wf1:.4f}")
    print(f"  delta vs exp09: {routed_wf1 - exp09_wf1:+.4f}")

    n_changed = (routed_preds != exp09_preds).sum()
    n_correct_before = (exp09_preds[routed_preds != exp09_preds] == labels[routed_preds != exp09_preds]).sum()
    n_correct_after = (routed_preds[routed_preds != exp09_preds] == labels[routed_preds != exp09_preds]).sum()
    print(f"\n  Rows where routing changed the prediction: {n_changed}")
    print(f"  Of those, exp09 was right: {n_correct_before}, routed model is right: {n_correct_after}")

    np.save(os.path.join(RESULTS_DIR, "confidence_routed_train_oof_preds.npy"), routed_preds)


if __name__ == "__main__":
    main()
