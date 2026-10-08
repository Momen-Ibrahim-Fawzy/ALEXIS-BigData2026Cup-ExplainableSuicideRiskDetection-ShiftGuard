"""
4-way version of ensemble_multi_source_3way.py — adds the dual-encoder
model (v14) as a 4th source alongside the three factor-gpt/factor_all/
factor_verified split-ensembles. The dual-encoder is a single unified
model (no vanilla/weighted split — DecoupledSoftmax already handles
co-occurring-positive interference differently from what the split-
ensemble addresses), so it contributes its own oof_probs/leaderboard_probs
directly rather than going through build_combined.

Still cheap (pure numpy, no GPU/training needed for the ensembling step
itself) and still just ONE threshold-tuning pass on the final averaged
matrix.

Usage:
    python ensemble_multi_source_4way.py
"""

import os, json
import numpy as np
import pandas as pd
from config import *
from predict_split_ensemble import load_probs, compute_pos_weights, build_combined, tune_thresholds

BASE = os.path.dirname(__file__)
SPLIT_ENSEMBLE_SOURCES = {
    "factor_gpt":      (os.path.join(BASE, "outputs_v6_weighted"),  os.path.join(BASE, "outputs_v6_vanilla")),
    "factor_all":      (os.path.join(BASE, "outputs_v8_weighted"),  os.path.join(BASE, "outputs_v8_vanilla")),
    "factor_verified": (os.path.join(BASE, "outputs_v10_weighted"), os.path.join(BASE, "outputs_v10_vanilla")),
}
DUAL_ENCODER_DIR = os.path.join(BASE, "outputs_v14_dual_encoder")


def main():
    pw = compute_pos_weights()
    labels = load_probs(SPLIT_ENSEMBLE_SOURCES["factor_gpt"][0], "oof_labels.npy")

    combined_oofs, combined_lbs, individual_f1s = {}, {}, {}
    for name, (w_dir, v_dir) in SPLIT_ENSEMBLE_SOURCES.items():
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

    # Dual-encoder: single model, no split-ensemble step
    de_oof = load_probs(DUAL_ENCODER_DIR, "oof_probs.npy")
    de_lb = load_probs(DUAL_ENCODER_DIR, "leaderboard_probs.npy")
    de_labels = load_probs(DUAL_ENCODER_DIR, "oof_labels.npy")
    assert np.array_equal(de_labels, labels), "Dual-encoder OOF labels must align with the split-ensemble sources' row order"
    combined_oofs["dual_encoder"] = de_oof
    combined_lbs["dual_encoder"] = de_lb
    _, de_f1s = tune_thresholds(de_oof, labels)
    individual_f1s["dual_encoder"] = np.mean(de_f1s)
    print(f"{'dual_encoder':16s} tuned Macro F1: {individual_f1s['dual_encoder']:.4f}")

    # 4-way average + ONE threshold-tuning pass
    ens_oof = np.mean(list(combined_oofs.values()), axis=0)
    ens_lb  = np.mean(list(combined_lbs.values()), axis=0)
    thresholds, ens_f1s = tune_thresholds(ens_oof, labels)
    ens_macro = np.mean(ens_f1s)

    # Compare against the already-confirmed 3-way (gpt+all+verified, no dual-encoder)
    three_way_oof = np.mean([combined_oofs["factor_gpt"], combined_oofs["factor_all"], combined_oofs["factor_verified"]], axis=0)
    _, three_way_f1s = tune_thresholds(three_way_oof, labels)
    three_way_macro = np.mean(three_way_f1s)

    print(f"\n3-way (gpt+all+verified, no dual-encoder) tuned Macro F1: {three_way_macro:.4f}  (OOF was 0.6020, submitted as v13)")
    print(f"4-way (+ dual_encoder) tuned Macro F1: {ens_macro:.4f}")
    print(f"4-way improvement over 3-way: {ens_macro - three_way_macro:+.4f}")

    result = {
        "individual_macro_f1": {k: float(v) for k, v in individual_f1s.items()},
        "three_way_macro_f1": float(three_way_macro),
        "four_way_macro_f1": float(ens_macro),
        "four_way_vs_three_way": float(ens_macro - three_way_macro),
    }
    with open(os.path.join(BASE, "sequence_data", "multi_source_4way_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if ens_macro <= three_way_macro:
        print("\n*** 4-way ensemble did NOT beat the already-confirmed 3-way ensemble on true OOF. "
              "Stick with the 3-way version (already submitted as v13). ***")
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

    out_path = os.path.join(BASE, "factor_predictions_multi_source_4way.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
