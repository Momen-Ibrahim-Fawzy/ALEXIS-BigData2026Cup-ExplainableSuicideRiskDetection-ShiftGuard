"""
3-way version of ensemble_multi_source.py — the 2-way average of factor_all
+ factor_verified was the biggest validated win this session (OOF +0.0176,
leaderboard +0.0276, even better than OOF predicted). This extends the same
idea to all three oversample-source variants we have fully trained
(factor_gpt from v6, factor_all from v8, factor_verified from v10),
following the same "ensembling diverse models reduces variance" logic.

Still cheap (pure numpy, no GPU/training) and still just ONE threshold-tuning
pass on the final averaged matrix — same low-overfitting-risk discipline.

Usage:
    python ensemble_multi_source_3way.py
"""

import os, json
import numpy as np
import pandas as pd
from config import *
from predict_split_ensemble import load_probs, compute_pos_weights, build_combined, tune_thresholds

BASE = os.path.dirname(__file__)
SOURCES = {
    "factor_gpt":      (os.path.join(BASE, "outputs_v6_weighted"),  os.path.join(BASE, "outputs_v6_vanilla")),
    "factor_all":      (os.path.join(BASE, "outputs_v8_weighted"),  os.path.join(BASE, "outputs_v8_vanilla")),
    "factor_verified": (os.path.join(BASE, "outputs_v10_weighted"), os.path.join(BASE, "outputs_v10_vanilla")),
}


def main():
    pw = compute_pos_weights()
    labels = load_probs(SOURCES["factor_gpt"][0], "oof_labels.npy")

    combined_oofs, combined_lbs, individual_f1s = {}, {}, {}
    for name, (w_dir, v_dir) in SOURCES.items():
        w_oof = load_probs(w_dir, "oof_probs.npy")
        v_oof = load_probs(v_dir, "oof_probs.npy")
        w_lb  = load_probs(w_dir, "leaderboard_probs.npy")
        v_lb  = load_probs(v_dir, "leaderboard_probs.npy")
        c_oof, _ = build_combined(w_oof, v_oof, pw)
        c_lb, _  = build_combined(w_lb, v_lb, pw)
        combined_oofs[name] = c_oof
        combined_lbs[name] = c_lb
        _, f1s = tune_thresholds(c_oof, labels)
        individual_f1s[name] = np.mean(f1s)
        print(f"{name:16s} tuned Macro F1: {individual_f1s[name]:.4f}")

    # 3-way average + ONE threshold-tuning pass
    ens_oof = np.mean(list(combined_oofs.values()), axis=0)
    ens_lb  = np.mean(list(combined_lbs.values()), axis=0)
    thresholds, ens_f1s = tune_thresholds(ens_oof, labels)
    ens_macro = np.mean(ens_f1s)

    best_individual = max(individual_f1s.values())
    # Also compare against the already-submitted, already-confirmed 2-way (all+verified) ensemble
    two_way_oof = 0.5 * (combined_oofs["factor_all"] + combined_oofs["factor_verified"])
    _, two_way_f1s = tune_thresholds(two_way_oof, labels)
    two_way_macro = np.mean(two_way_f1s)

    print(f"\n2-way (all+verified) tuned Macro F1: {two_way_macro:.4f}  (confirmed on real leaderboard: 0.5799)")
    print(f"3-way (gpt+all+verified) tuned Macro F1: {ens_macro:.4f}")
    print(f"Best individual source: {best_individual:.4f}")
    print(f"3-way improvement over 2-way: {ens_macro - two_way_macro:+.4f}")
    print(f"3-way improvement over best individual: {ens_macro - best_individual:+.4f}")

    result = {
        "individual_macro_f1": {k: float(v) for k, v in individual_f1s.items()},
        "two_way_macro_f1": float(two_way_macro),
        "three_way_macro_f1": float(ens_macro),
        "three_way_vs_two_way": float(ens_macro - two_way_macro),
    }
    with open(os.path.join(BASE, "sequence_data", "multi_source_3way_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if ens_macro <= two_way_macro:
        print("\n*** 3-way ensemble did NOT beat the already-confirmed 2-way ensemble on true OOF. "
              "Stick with the 2-way version (already submitted as v12). ***")
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

    out_path = os.path.join(BASE, "factor_predictions_multi_source_3way.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
