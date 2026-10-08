"""
Extends ensemble_multi_source_4way.py to optionally include the v15
context-aware model (Sub1a risk-level + has_evidence injected as an
auxiliary feature) as a 5th source, alongside v14's dual-encoder. Both v14
and v15 are OPTIONAL here — this script builds the best ensemble from
whichever of {3-way split-ensemble sources, dual_encoder, context} are
available, and always compares against the best previously-confirmed
result (3-way OOF 0.6020, submitted as v13) before deciding whether to
write a new submission.

Usage:
    python ensemble_multi_source_5way.py
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
CONTEXT_DIR = os.path.join(BASE, "outputs_v15_context")
PRIOR_CONFIRMED_BEST = 0.6020  # v13's 3-way ensemble, true OOF, submitted and real-scored


def available(dir_path):
    return os.path.exists(os.path.join(dir_path, "oof_probs.npy")) and \
           os.path.exists(os.path.join(dir_path, "leaderboard_probs.npy"))


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

    extra_sources = {"dual_encoder": DUAL_ENCODER_DIR, "context": CONTEXT_DIR}
    included_extra = []
    for name, d in extra_sources.items():
        if available(d):
            oof = load_probs(d, "oof_probs.npy")
            lb = load_probs(d, "leaderboard_probs.npy")
            src_labels = load_probs(d, "oof_labels.npy")
            assert np.array_equal(src_labels, labels), f"{name}'s OOF labels must align with the split-ensemble sources' row order"
            combined_oofs[name] = oof
            combined_lbs[name] = lb
            _, f1s = tune_thresholds(oof, labels)
            individual_f1s[name] = np.mean(f1s)
            print(f"{name:16s} tuned Macro F1: {individual_f1s[name]:.4f}")
            included_extra.append(name)
        else:
            print(f"{name:16s} not available yet — skipping")

    ens_oof = np.mean(list(combined_oofs.values()), axis=0)
    ens_lb  = np.mean(list(combined_lbs.values()), axis=0)
    thresholds, ens_f1s = tune_thresholds(ens_oof, labels)
    ens_macro = np.mean(ens_f1s)

    n_sources = 3 + len(included_extra)
    print(f"\n{n_sources}-way ensemble (3-way + {'+'.join(included_extra) if included_extra else 'none'}) tuned Macro F1: {ens_macro:.4f}")
    print(f"Prior confirmed best (v13, 3-way, real leaderboard-validated): {PRIOR_CONFIRMED_BEST:.4f}")
    print(f"Improvement: {ens_macro - PRIOR_CONFIRMED_BEST:+.4f}")

    result = {
        "individual_macro_f1": {k: float(v) for k, v in individual_f1s.items()},
        "included_extra_sources": included_extra,
        "ensemble_macro_f1": float(ens_macro),
        "prior_confirmed_best": PRIOR_CONFIRMED_BEST,
        "improvement": float(ens_macro - PRIOR_CONFIRMED_BEST),
    }
    with open(os.path.join(BASE, "sequence_data", "multi_source_5way_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if ens_macro <= PRIOR_CONFIRMED_BEST:
        print("\n*** Did NOT beat the prior confirmed best (v13's 3-way ensemble) on true OOF. "
              "Do not submit — stick with v13/v14 as applicable. ***")
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

    out_path = os.path.join(BASE, f"factor_predictions_multi_source_{n_sources}way.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
