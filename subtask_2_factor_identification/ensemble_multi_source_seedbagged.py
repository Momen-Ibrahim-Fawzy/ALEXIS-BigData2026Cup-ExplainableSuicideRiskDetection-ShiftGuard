"""Same as ensemble_multi_source.py, but factor_verified's weighted half is
the 3-way seed-bagged model (outputs_v10_weighted_seedbag, true OOF tuned
Macro F1 0.5977 combined with vanilla) instead of the single-seed one
(outputs_v10_weighted, 0.5806) -- tests whether seed-bagging's gain STACKS
with the existing factor_all+factor_verified cross-source ensemble (current
best: 0.5982), rather than just replacing it.

Usage:
    python ensemble_multi_source_seedbagged.py
"""

import os, json
import numpy as np
import pandas as pd
from config import *
from predict_split_ensemble import load_probs, compute_pos_weights, build_combined, tune_thresholds

BASE = os.path.dirname(__file__)
FACTOR_ALL_W = os.path.join(BASE, "outputs_v8_weighted")
FACTOR_ALL_V = os.path.join(BASE, "outputs_v8_vanilla")
FACTOR_VER_W = os.path.join(BASE, "outputs_v10_weighted_seedbag")  # <-- seed-bagged, was outputs_v10_weighted
FACTOR_VER_V = os.path.join(BASE, "outputs_v10_vanilla")


def main():
    pw = compute_pos_weights()

    w_oof_all = load_probs(FACTOR_ALL_W, "oof_probs.npy")
    v_oof_all = load_probs(FACTOR_ALL_V, "oof_probs.npy")
    labels    = load_probs(FACTOR_ALL_W, "oof_labels.npy")
    w_lb_all  = load_probs(FACTOR_ALL_W, "leaderboard_probs.npy")
    v_lb_all  = load_probs(FACTOR_ALL_V, "leaderboard_probs.npy")
    combined_oof_all, _ = build_combined(w_oof_all, v_oof_all, pw)
    combined_lb_all, _  = build_combined(w_lb_all, v_lb_all, pw)

    w_oof_ver = load_probs(FACTOR_VER_W, "oof_probs.npy")
    v_oof_ver = load_probs(FACTOR_VER_V, "oof_probs.npy")
    w_lb_ver  = load_probs(FACTOR_VER_W, "leaderboard_probs.npy")
    v_lb_ver  = load_probs(FACTOR_VER_V, "leaderboard_probs.npy")
    combined_oof_ver, _ = build_combined(w_oof_ver, v_oof_ver, pw)
    combined_lb_ver, _  = build_combined(w_lb_ver, v_lb_ver, pw)

    _, base_all_f1s = tune_thresholds(combined_oof_all, labels)
    _, base_ver_f1s = tune_thresholds(combined_oof_ver, labels)
    print(f"factor_all                 tuned Macro F1: {np.mean(base_all_f1s):.4f}")
    print(f"factor_verified (seedbag)  tuned Macro F1: {np.mean(base_ver_f1s):.4f}")

    ens_oof = 0.5 * (combined_oof_all + combined_oof_ver)
    ens_lb  = 0.5 * (combined_lb_all + combined_lb_ver)
    thresholds, ens_f1s = tune_thresholds(ens_oof, labels)
    ens_macro = np.mean(ens_f1s)
    print(f"\n2-way ensemble (with seed-bagged verified)  tuned Macro F1: {ens_macro:.4f}")

    best_individual = max(np.mean(base_all_f1s), np.mean(base_ver_f1s))
    print(f"Best individual source:        {best_individual:.4f}")
    print(f"Improvement over best individual: {ens_macro - best_individual:+.4f}")
    print(f"vs current best (single-seed 2-way ensemble): 0.5982")
    print(f"Delta vs current best: {ens_macro - 0.5982:+.4f}")

    result = {
        "factor_all_macro_f1": float(np.mean(base_all_f1s)),
        "factor_verified_seedbagged_macro_f1": float(np.mean(base_ver_f1s)),
        "ensemble_macro_f1": float(ens_macro),
        "improvement_over_best_individual": float(ens_macro - best_individual),
        "improvement_over_current_best_2way": float(ens_macro - 0.5982),
    }
    with open(os.path.join(BASE, "sequence_data", "multi_source_ensemble_seedbagged_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if ens_macro <= 0.5982:
        print("\n*** Did NOT beat the current best 2-way ensemble (0.5982). "
              "Seed-bagging's gain does not stack additively here. ***")
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

    out_path = os.path.join(BASE, "factor_predictions_multi_source_seedbagged.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
