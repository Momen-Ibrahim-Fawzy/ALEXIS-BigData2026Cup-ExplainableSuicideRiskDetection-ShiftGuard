"""
2-way Sub1a ensemble: exp09 (DeBERTa-v3-large, true OOF WF1 0.8014) +
exp14 (AIMH/mental-roberta-large, true OOF WF1 0.7962) — similar individual
strength, architecturally diverse, same profile that made the Sub2
factor_all+factor_verified ensemble work. Averages softmax probabilities
(logit scales differ across architectures, so probability-space averaging
is the fair combination, not raw logit averaging).

Validates on true OOF (both experiments' OOF logits come from held-out
folds only) before touching the leaderboard set.

Usage:
    python extract_logits.py            # exp09, already done
    python extract_logits_exp14.py      # exp14
    python ensemble_exp09_exp14.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK2ID, ID2RISK, RESULTS_DIR, BASE_DIR

EXP09_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
EXP14_DIR = os.path.join(RESULTS_DIR, "exp14_mental_roberta_large", "sequence_data")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    train_meta = pd.read_csv(os.path.join(EXP09_DIR, "train_meta.csv"))
    exp09_oof = np.load(os.path.join(EXP09_DIR, "train_oof_logits.npy"))
    exp14_oof = np.load(os.path.join(EXP14_DIR, "train_oof_logits.npy"))
    true_labels = np.array([RISK2ID[r] for r in train_meta["risk_norm"]])

    exp09_probs = softmax(exp09_oof)
    exp14_probs = softmax(exp14_oof)

    wf1_09 = f1_score(true_labels, exp09_probs.argmax(1), average="weighted", zero_division=0)
    wf1_14 = f1_score(true_labels, exp14_probs.argmax(1), average="weighted", zero_division=0)
    print(f"exp09 alone   true OOF WF1={wf1_09:.4f}")
    print(f"exp14 alone   true OOF WF1={wf1_14:.4f}")

    best_w, best_wf1 = None, -1
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * exp09_probs + (1 - w) * exp14_probs
        wf1 = f1_score(true_labels, blend.argmax(1), average="weighted", zero_division=0)
        if wf1 > best_wf1:
            best_wf1, best_w = wf1, w
    print(f"\nBest blend weight (exp09 share): {best_w:.2f}  true OOF WF1={best_wf1:.4f}")

    blend_50 = 0.5 * exp09_probs + 0.5 * exp14_probs
    wf1_50 = f1_score(true_labels, blend_50.argmax(1), average="weighted", zero_division=0)
    print(f"Plain 50/50 blend            true OOF WF1={wf1_50:.4f}")

    print(f"\n(compare to exp09 alone 0.8014, exp09+seq-rescored 0.8078)")

    result = {
        "exp09_alone_wf1": float(wf1_09),
        "exp14_alone_wf1": float(wf1_14),
        "blend_50_50_wf1": float(wf1_50),
        "best_blend_weight_exp09": float(best_w),
        "best_blend_wf1": float(best_wf1),
    }
    out_dir = os.path.join(RESULTS_DIR, "ensemble_exp09_exp14")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "oof_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if best_wf1 <= 0.8078:
        print("\n*** Ensemble does NOT beat the already-validated exp09+seq-rescore "
              "(0.8078) on true OOF. Not worth pursuing further as-is. ***")
        return

    # Save blended OOF logits (as log-probs, so downstream seq-rescoring can
    # still operate in a logit-like space) for a follow-up: does sequence
    # rescoring stacked on TOP of this blend do even better?
    blended_oof = best_w * exp09_probs + (1 - best_w) * exp14_probs
    np.save(os.path.join(out_dir, "train_oof_blend_probs.npy"), blended_oof)
    train_meta.to_csv(os.path.join(out_dir, "train_meta.csv"), index=False)

    exp09_lb = np.load(os.path.join(EXP09_DIR, "leaderboard_logits.npy"))
    exp14_lb = np.load(os.path.join(EXP14_DIR, "leaderboard_logits.npy"))
    lb_meta = pd.read_csv(os.path.join(EXP09_DIR, "leaderboard_meta.csv"))
    lb_blend = best_w * softmax(exp09_lb) + (1 - best_w) * softmax(exp14_lb)
    np.save(os.path.join(out_dir, "leaderboard_blend_probs.npy"), lb_blend)
    lb_meta.to_csv(os.path.join(out_dir, "leaderboard_meta.csv"), index=False)
    print(f"\nSaved blended OOF + leaderboard probs -> {out_dir}")


if __name__ == "__main__":
    main()
