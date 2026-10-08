"""Merge partial fold OOF outputs from train_nli_ordinal_kfold.py, compute
the true 5-fold OOF WF1 (no threshold tuning anywhere in this pipeline -- the
0.5 cumulative-threshold decode is fixed -- so this number is already fully
honest, same discipline as this project's other Sub1a scripts), and report
it against the two relevant baselines: exp09 alone (0.8014, apples-to-apples
architecture comparison) and production with sequence-rescoring stacked on
top (0.8078, the actual number sitting in the current submission).

Usage:
    python merge_nli_ordinal_folds.py --output-dir outputs_nli_ordinal
"""

import argparse, glob, json, os
import numpy as np
import pandas as pd

from risk_hypotheses import RISK_LEVELS
from train_nli_ordinal_kfold import compute_wf1

EXP09_BASELINE_WF1 = 0.8014       # exp09 alone, true OOF (EXPERIMENT_REPORT.md)
PRODUCTION_WF1 = 0.8078            # exp09 + sequence-rescoring, true OOF (currently deployed)

PSEUDO_LB_SPLIT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pseudo_leaderboard_users.json")  # baselines/pseudo_leaderboard_users.json -- shared across subtasks,
                                        # relocated out of subtask_2_factor_identification (deleted 2026-07-25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    output_dir = os.path.abspath(args.output_dir)

    partial_probs_paths = sorted(glob.glob(os.path.join(output_dir, "oof_probs_folds_*.npy")))
    partial_filled_paths = sorted(glob.glob(os.path.join(output_dir, "oof_filled_folds_*.npy")))
    if not partial_probs_paths:
        raise FileNotFoundError(f"No oof_probs_folds_*.npy found in {output_dir}")
    print(f"Merging {len(partial_probs_paths)} partial file(s): {[os.path.basename(p) for p in partial_probs_paths]}")

    oof_probs, oof_filled = None, None
    for pp, fp in zip(partial_probs_paths, partial_filled_paths):
        p, f = np.load(pp), np.load(fp)
        if oof_probs is None:
            oof_probs = np.zeros_like(p)
            oof_filled = np.zeros_like(f)
        overlap = oof_filled & f
        if overlap.any():
            raise ValueError(f"{overlap.sum()} rows filled by more than one partial file -- refusing to merge.")
        oof_probs[f] = p[f]
        oof_filled |= f

    if not oof_filled.all():
        raise ValueError(f"{(~oof_filled).sum()} rows never filled -- check --fold-ids coverage across runs.")

    oof_labels = np.load(os.path.join(output_dir, "oof_labels.npy"))
    np.save(os.path.join(output_dir, "oof_probs.npy"), oof_probs)

    wf1 = compute_wf1(oof_probs, oof_labels)
    print(f"\nTrue 5-fold OOF WF1 (ordinal cumulative-NLI, fixed 0.5 thresholds): {wf1:.4f}")
    print(f"Compare to: exp09 alone (true OOF) {EXP09_BASELINE_WF1:.4f}  |  "
          f"production w/ seq-rescoring (true OOF) {PRODUCTION_WF1:.4f}")

    result = {"true_oof_wf1": float(wf1), "exp09_baseline": EXP09_BASELINE_WF1,
              "production_with_seqrescore": PRODUCTION_WF1}

    if os.path.exists(PSEUDO_LB_SPLIT_PATH):
        with open(PSEUDO_LB_SPLIT_PATH) as f:
            holdout_users = set(json.load(f)["pseudo_leaderboard_users"])
        meta = pd.read_csv(os.path.join(output_dir, "oof_meta.csv"))
        is_holdout = meta["anon_user_id"].isin(holdout_users).values
        wf1_holdout = compute_wf1(oof_probs[is_holdout], oof_labels[is_holdout])
        print(f"(context only, no tuning involved) WF1 on Sub2's fixed 20 pseudo-leaderboard users: "
              f"{wf1_holdout:.4f} ({is_holdout.sum()} rows)")
        result["wf1_on_pseudo_lb_users_context_only"] = float(wf1_holdout)

    with open(os.path.join(output_dir, "ordinal_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if wf1 > EXP09_BASELINE_WF1:
        print("\n*** BEATS exp09 alone -- the ordinal-NLI reformulation is intrinsically stronger. ***")
    else:
        print("\n*** Did NOT beat exp09 alone. ***")
    if wf1 > PRODUCTION_WF1:
        print("*** ALSO beats production (with seq-rescoring) -- worth a real submission. ***")
    else:
        print("*** Does not (yet) beat production's deployed number. ***")


if __name__ == "__main__":
    main()
