"""
Average the two fully-trained Sub2 oversample-source variants (factor_all
from v8, factor_verified from v10) rather than picking just one. Cheap
(pure numpy on already-computed OOF/leaderboard arrays, no GPU/training
needed) and low overfitting risk — a single 2-way average plus ONE
principled threshold-tuning pass, not a per-factor grid search over many
options (which is what made the earlier Sub2 sequence-rescoring attempt
overfit on small per-fold samples).

Usage:
    python ensemble_multi_source.py
"""

import os, json
import numpy as np
import pandas as pd
from config import *
from predict_split_ensemble import load_probs, compute_pos_weights, build_combined, tune_thresholds

BASE = os.path.dirname(__file__)
FACTOR_ALL_W = os.path.join(BASE, "outputs_v8_weighted")
FACTOR_ALL_V = os.path.join(BASE, "outputs_v8_vanilla")
FACTOR_VER_W = os.path.join(BASE, "outputs_v10_weighted")
FACTOR_VER_V = os.path.join(BASE, "outputs_v10_vanilla")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(BASE), "submissions")


def main():
    pw = compute_pos_weights()

    # factor_all's own combined (vanilla/weighted split-ensemble) matrices
    w_oof_all = load_probs(FACTOR_ALL_W, "oof_probs.npy")
    v_oof_all = load_probs(FACTOR_ALL_V, "oof_probs.npy")
    labels    = load_probs(FACTOR_ALL_W, "oof_labels.npy")
    w_lb_all  = load_probs(FACTOR_ALL_W, "leaderboard_probs.npy")
    v_lb_all  = load_probs(FACTOR_ALL_V, "leaderboard_probs.npy")
    combined_oof_all, _ = build_combined(w_oof_all, v_oof_all, pw)
    combined_lb_all, _  = build_combined(w_lb_all, v_lb_all, pw)

    # factor_verified's own combined matrices
    w_oof_ver = load_probs(FACTOR_VER_W, "oof_probs.npy")
    v_oof_ver = load_probs(FACTOR_VER_V, "oof_probs.npy")
    w_lb_ver  = load_probs(FACTOR_VER_W, "leaderboard_probs.npy")
    v_lb_ver  = load_probs(FACTOR_VER_V, "leaderboard_probs.npy")
    combined_oof_ver, _ = build_combined(w_oof_ver, v_oof_ver, pw)
    combined_lb_ver, _  = build_combined(w_lb_ver, v_lb_ver, pw)

    # Baselines (already known, recomputed here for a clean side-by-side)
    _, base_all_f1s = tune_thresholds(combined_oof_all, labels)
    _, base_ver_f1s = tune_thresholds(combined_oof_ver, labels)
    print(f"factor_all      tuned Macro F1: {np.mean(base_all_f1s):.4f}")
    print(f"factor_verified tuned Macro F1: {np.mean(base_ver_f1s):.4f}")

    # 2-way average + ONE threshold-tuning pass on the averaged OOF matrix
    ens_oof = 0.5 * (combined_oof_all + combined_oof_ver)
    ens_lb  = 0.5 * (combined_lb_all + combined_lb_ver)
    thresholds, ens_f1s = tune_thresholds(ens_oof, labels)
    ens_macro = np.mean(ens_f1s)
    print(f"\n2-way ensemble  tuned Macro F1: {ens_macro:.4f}")

    best_individual = max(np.mean(base_all_f1s), np.mean(base_ver_f1s))
    print(f"Best individual source:        {best_individual:.4f}")
    print(f"Improvement over best individual: {ens_macro - best_individual:+.4f}")

    result = {
        "factor_all_macro_f1": float(np.mean(base_all_f1s)),
        "factor_verified_macro_f1": float(np.mean(base_ver_f1s)),
        "ensemble_macro_f1": float(ens_macro),
        "improvement_over_best_individual": float(ens_macro - best_individual),
    }
    with open(os.path.join(BASE, "sequence_data", "multi_source_ensemble_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if ens_macro <= best_individual:
        print("\n*** 2-way ensemble did NOT beat the best individual source on true OOF. "
              "Do not use it for a submission. ***")
        return

    preds = (ens_lb >= thresholds).astype(int)
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    rows = []
    for i in range(len(lb_df)):
        facs = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j] == 1]
        rows.append({"row_id": lb_df.iloc[i]["row_id"], "factors": str(facs) if facs else "[]"})
    out = pd.DataFrame(rows)
    n = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"Mean factors/post: {n.mean():.2f}")

    out_path = os.path.join(BASE, "factor_predictions_multi_source_ensemble.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
