"""Subtask 2: Factor-level split ensemble of vanilla-BCE and weighted-BCE models.

Assigns each factor to the model that handles it better based on class imbalance:

  neg/pos ≤ 5   → vanilla-BCE model (outputs_vanilla/)
                   Common factors: high pos count, both models have signal,
                   but vanilla preserves precision better (no representation bleed
                   from high-weight rare factors).

  neg/pos > 10  → weighted-BCE model (outputs/)
                   Rare factors: vanilla BCE ignores them entirely; only the
                   reweighted model has non-trivial recall.

  5 < neg/pos ≤ 10 → average of both models (medium zone)

After building the combined OOF probability matrix, per-factor thresholds are
re-tuned on those combined probs, then applied to the combined leaderboard probs.

Prerequisites:
  outputs/oof_probs.npy               — v2 weighted-BCE OOF (from train.py)
  outputs/oof_labels.npy              — same labels for both models
  outputs/leaderboard_probs.npy       — v2 weighted-BCE leaderboard (from predict.py)
  outputs_vanilla/oof_probs.npy       — vanilla-BCE OOF (from this run)
  outputs_vanilla/leaderboard_probs.npy — vanilla-BCE leaderboard (from predict.py --output-dir)

Writes:
  outputs/factor_thresholds.json      — re-tuned on combined OOF (overwrites v2 thresholds)
  outputs/factor_predictions.csv      — final predictions (what merge_submission.py reads)
"""

import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from config import *

# Defaults (can be overridden via CLI args in main())
VANILLA_DIR  = os.path.join(os.path.dirname(__file__), "outputs_vanilla")
WEIGHTED_DIR = OUTPUT_DIR   # outputs/ from config

# Factor split boundaries (pos_weight = neg/pos on full train set)
PW_COMMON = 5.0   # pos_weight ≤ this → use vanilla
PW_RARE   = 10.0  # pos_weight > this → use weighted


def load_probs(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return np.load(path)


def compute_pos_weights():
    """Compute neg/pos ratio per factor from full training set."""
    import pandas as pd_inner

    xlsx = os.path.join(os.path.dirname(__file__), "data", "train.xlsx")
    df = pd_inner.read_excel(xlsx, engine="openpyxl")

    def parse(x):
        if isinstance(x, str) and x.startswith("["):
            try: return eval(x)
            except: pass
        if isinstance(x, str): return [f.strip() for f in x.split(",") if f.strip()]
        return []

    df["factors_list"] = df["factors"].apply(parse)
    N = len(df)
    pos = np.zeros(NUM_FACTORS)
    for _, row in df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                pos[FACTOR2ID[fl]] += 1
    neg = N - pos
    return neg / (pos + 1e-8)


def build_combined(weighted_probs, vanilla_probs, pw):
    """Build combined probability matrix with per-factor model selection."""
    combined = np.empty_like(weighted_probs)
    source = []
    for i in range(NUM_FACTORS):
        if pw[i] <= PW_COMMON:
            combined[:, i] = vanilla_probs[:, i]
            source.append("vanilla")
        elif pw[i] > PW_RARE:
            combined[:, i] = weighted_probs[:, i]
            source.append("weighted")
        else:
            combined[:, i] = 0.5 * (vanilla_probs[:, i] + weighted_probs[:, i])
            source.append("average")
    return combined, source


def tune_thresholds(probs, labels):
    """Per-factor threshold sweep; returns (thresholds, per_factor_f1)."""
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    tuned_f1s = []
    for i in range(NUM_FACTORS):
        best_f1, best_t = 0.0, 0.5
        for t in candidates:
            f1 = f1_score(labels[:, i], (probs[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
        tuned_f1s.append(best_f1)
    return thresholds, tuned_f1s


def main():
    global VANILLA_DIR, WEIGHTED_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighted-dir", default=None,
                        help="Directory with weighted-BCE outputs (default: outputs/)")
    parser.add_argument("--vanilla-dir", default=None,
                        help="Directory with vanilla-BCE outputs (default: outputs_vanilla/)")
    args = parser.parse_args()
    if args.weighted_dir:
        WEIGHTED_DIR = os.path.abspath(args.weighted_dir)
    if args.vanilla_dir:
        VANILLA_DIR = os.path.abspath(args.vanilla_dir)

    print("=" * 64)
    print("  Sub2 Factor-Level Split Ensemble")
    print(f"  Vanilla dir : {VANILLA_DIR}")
    print(f"  Weighted dir: {WEIGHTED_DIR}")
    print("=" * 64)

    # Load probabilities
    w_oof   = load_probs(WEIGHTED_DIR, "oof_probs.npy")
    v_oof   = load_probs(VANILLA_DIR,  "oof_probs.npy")
    labels  = load_probs(WEIGHTED_DIR, "oof_labels.npy")   # same for both
    w_lb    = load_probs(WEIGHTED_DIR, "leaderboard_probs.npy")
    v_lb    = load_probs(VANILLA_DIR,  "leaderboard_probs.npy")

    assert w_oof.shape == v_oof.shape == labels.shape, "OOF shape mismatch"
    assert w_lb.shape  == v_lb.shape, "Leaderboard shape mismatch"
    print(f"OOF shape: {w_oof.shape}  |  LB shape: {w_lb.shape}")

    # Compute pos_weights from training data
    pw = compute_pos_weights()

    # Build combined matrices
    oof_combined, sources = build_combined(w_oof, v_oof, pw)
    lb_combined,  _       = build_combined(w_lb,  v_lb,  pw)

    # Per-factor OOF comparison table
    print(f"\n{'Factor':42s} {'PW':>6} {'Src':>8} {'W-F1':>6} {'V-F1':>6} {'Ens-F1':>7}")
    print("-" * 76)

    default_w_f1s, default_v_f1s, default_ens_f1s = [], [], []
    for i, f in enumerate(FACTOR_LIST):
        wf1 = f1_score(labels[:, i], (w_oof[:, i] >= 0.5).astype(int), zero_division=0)
        vf1 = f1_score(labels[:, i], (v_oof[:, i] >= 0.5).astype(int), zero_division=0)
        ef1 = f1_score(labels[:, i], (oof_combined[:, i] >= 0.5).astype(int), zero_division=0)
        default_w_f1s.append(wf1); default_v_f1s.append(vf1); default_ens_f1s.append(ef1)
        marker = " <" if ef1 > max(wf1, vf1) + 0.01 else ""
        print(f"  {f:40s} {pw[i]:6.1f} {sources[i]:>8} {wf1:6.3f} {vf1:6.3f} {ef1:7.3f}{marker}")

    print(f"\n  Default (t=0.5) Macro F1 — Weighted: {np.mean(default_w_f1s):.4f} | "
          f"Vanilla: {np.mean(default_v_f1s):.4f} | Ensemble: {np.mean(default_ens_f1s):.4f}")

    # Tune thresholds on combined OOF
    thresholds, tuned_f1s = tune_thresholds(oof_combined, labels)
    tuned_macro = np.mean(tuned_f1s)
    default_macro = np.mean(default_ens_f1s)

    print(f"\n  Tuned Macro F1 (combined OOF): {tuned_macro:.4f}  "
          f"(+{tuned_macro - default_macro:.4f} vs default t=0.5)")

    # Save thresholds
    thresh_result = {
        "default_macro_f1": float(default_macro),
        "tuned_macro_f1":   float(tuned_macro),
        "improvement":      float(tuned_macro - default_macro),
        "thresholds":       {FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)},
        "per_factor_f1":    {FACTOR_LIST[i]: float(tuned_f1s[i])  for i in range(NUM_FACTORS)},
        "model_source":     {FACTOR_LIST[i]: sources[i]           for i in range(NUM_FACTORS)},
    }
    thresh_path = os.path.join(WEIGHTED_DIR, "factor_thresholds.json")
    with open(thresh_path, "w") as fh:
        json.dump(thresh_result, fh, indent=2)
    print(f"  Thresholds saved → {thresh_path}")

    # Generate leaderboard predictions
    lb_df = pd.read_excel(
        os.path.join(os.path.dirname(__file__), "data", "leaderboard.xlsx"),
        engine="openpyxl"
    )
    preds = (lb_combined >= thresholds).astype(int)

    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"],
                     "factors": str(factors) if factors else "[]"})

    out = pd.DataFrame(rows)
    out_path = os.path.join(WEIGHTED_DIR, "factor_predictions.csv")
    out.to_csv(out_path, index=False)

    out["n"] = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"\n  Saved → {out_path}")
    print(f"  Mean factors/post: {out['n'].mean():.1f}  |  "
          f"Zero-factor posts: {(out['n']==0).sum()}/{len(out)}")

    # Per-factor model source summary
    counts = {"vanilla": 0, "weighted": 0, "average": 0}
    for s in sources:
        counts[s] += 1
    print(f"\n  Model assignment: vanilla={counts['vanilla']} factors, "
          f"weighted={counts['weighted']} factors, average={counts['average']} factors")

    print(f"\n  Combined OOF Macro F1 (tuned): {tuned_macro:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
