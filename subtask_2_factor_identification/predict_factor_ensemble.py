"""Subtask 2: GPT-OSS + DeBERTa factor ensemble.

Combines DeBERTa leaderboard sigmoid probabilities (0.65 weight) with
GPT-OSS-120B binary factor predictions (0.35 weight) then applies the
per-factor tuned thresholds from tune_thresholds.py.

Prerequisites:
  outputs/leaderboard_probs.npy   — saved by predict.py (run it first)
  outputs/factor_thresholds.json  — saved by tune_thresholds.py
  data/LLM_Annotator/results/leaderboard_gpt-oss-120b.json
"""

import os, json
import numpy as np
import pandas as pd
from config import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_RESULTS = os.path.join(BASE_DIR, "data", "LLM_Annotator", "results", "leaderboard_gpt-oss-120b.json")
DEBERTA_W    = 0.65
LLM_W        = 0.35


def load_llm_binary(lb_df):
    with open(LLM_RESULTS) as f:
        llm_data = json.load(f)
    llm_map = {item["id"]: item["factors"] for item in llm_data}

    mat = np.zeros((len(lb_df), NUM_FACTORS))
    missing = 0
    for i, row in lb_df.iterrows():
        rid = row["row_id"]
        factors = llm_map.get(rid)
        if factors is None:
            missing += 1
            continue
        for f in factors:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                mat[i, FACTOR2ID[fl]] = 1.0
    if missing:
        print(f"  Warning: {missing} leaderboard rows had no LLM prediction")
    total = mat.sum()
    print(f"  LLM: {total:.0f} labels across {len(lb_df)} posts "
          f"(avg {total/len(lb_df):.1f}/post)")
    return mat


def main():
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")

    # Load DeBERTa leaderboard probabilities
    probs_path = os.path.join(OUTPUT_DIR, "leaderboard_probs.npy")
    if not os.path.exists(probs_path):
        raise FileNotFoundError(
            f"DeBERTa probs not found: {probs_path}\n"
            "Run predict.py first (it saves leaderboard_probs.npy)."
        )
    deberta_probs = np.load(probs_path)
    print(f"DeBERTa probs: {deberta_probs.shape}, "
          f"range [{deberta_probs.min():.3f}, {deberta_probs.max():.3f}]")

    # Load LLM binary predictions
    print("Loading GPT-OSS-120B predictions...")
    llm_binary = load_llm_binary(lb_df)

    # Weighted ensemble: both scales are compatible (DeBERTa is 0–1 sigmoid,
    # LLM is 0/1 binary — blending gives a soft score between 0 and 1)
    ensemble = DEBERTA_W * deberta_probs + LLM_W * llm_binary

    # Load thresholds tuned on DeBERTa OOF probs
    thresh_path = os.path.join(OUTPUT_DIR, "factor_thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            td = json.load(f)
        thresholds = np.array([td["thresholds"].get(f, 0.5) for f in FACTOR_LIST])
        print(f"Using tuned thresholds (DeBERTa OOF Macro F1={td['tuned_macro_f1']:.4f})")
    else:
        thresholds = np.full(NUM_FACTORS, 0.5)
        print("Using default thresholds (0.5)")

    preds = (ensemble >= thresholds).astype(int)

    # Build output
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"],
                     "factors": str(factors) if factors else "[]"})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "factor_predictions.csv")
    out.to_csv(out_path, index=False)

    out["n"] = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"\nSaved → {out_path}")
    print(f"Mean factors/post: {out['n'].mean():.1f}, "
          f"zero: {(out['n']==0).sum()}/{len(out)}")

    # Show where ensemble differs from DeBERTa-only
    deberta_preds = (deberta_probs >= thresholds).astype(int)
    llm_preds     = (llm_binary >= 0.5).astype(int)
    diff = preds.sum(0) - deberta_preds.sum(0)
    changed = [(FACTOR_LIST[i], int(deberta_preds[:, i].sum()),
                int(llm_preds[:, i].sum()), int(preds[:, i].sum()))
               for i in range(NUM_FACTORS) if diff[i] != 0]
    if changed:
        print("\nFactors changed by ensemble (DeBERTa → Ensemble):")
        print(f"  {'Factor':42s} {'DeBERTa':>8} {'LLM':>6} {'Ensemble':>9}")
        for fname, d, l, e in sorted(changed, key=lambda x: x[3]-x[1]):
            delta = "+" if e > d else ""
            print(f"  {fname:42s} {d:8d} {l:6d} {e:9d} ({delta}{e-d})")
    else:
        print("\nEnsemble produces identical predictions to DeBERTa-only at these thresholds.")


if __name__ == "__main__":
    main()
