"""Sub2 v2 leaderboard prediction: 5-fold ensemble + threshold tuning.

Threshold-tuning methodology deliberately changed from every prior Sub2
script: by default, thresholds are tuned ONLY on the ~133 non-pseudo-
leaderboard users' OOF, not the full 153. This project just has direct
evidence that tuning-and-scoring on the full 153 users is unreliable (v18:
naive full-OOF said +0.0083, real leaderboard showed -0.0326 -- a 0.041
gap), while the honest 133-tuned/20-scored check predicted the real result
to within 0.004. Given v2 is an even more elaborate, more tunable pipeline
than v18, defaulting to the methodology we have hard evidence overfits is
hard to justify. Pass --tune-on-full to opt into the old methodology anyway
(e.g. for a direct side-by-side comparison).

Needs Sub1a's leaderboard risk predictions (--risk-csv) for the special
risk-level token -- the model was trained expecting Sub1a's PREDICTED risk
at inference time, not ground truth (which doesn't exist for leaderboard
rows anyway).

Usage:
    python predict_v2.py --output-dir outputs_v2_factor_verified \\
        --risk-csv ../subtask_1a_risk_classification/submissions/submission_exp09_deberta_large_lr_high_seqrescored.csv
"""

import argparse, json, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from config import *
from dataset_v2 import FactorDatasetV2, load_sbert_lookup
from model_v2 import FactorClassifierV2, add_risk_tokens
from pseudo_leaderboard_split import get_or_create_split


def tune_thresholds(probs, labels):
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    for i in range(NUM_FACTORS):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def compute_macro_f1(probs, labels, thresholds):
    preds = (probs >= thresholds).astype(int)
    return float(np.mean([f1_score(labels[:, i], preds[:, i], zero_division=0) for i in range(NUM_FACTORS)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--risk-csv", required=True, help="Sub1a's leaderboard risk predictions CSV")
    ap.add_argument("--tune-on-full", action="store_true",
                     help="Use the old (demonstrated-fragile) full-153-user tuning instead of "
                          "the pseudo-leaderboard-safe 133-user tuning. Not recommended.")
    args = ap.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_probs = np.load(os.path.join(output_dir, "oof_probs.npy"))
    oof_labels = np.load(os.path.join(output_dir, "oof_labels.npy"))
    oof_meta = pd.read_csv(os.path.join(output_dir, "oof_meta.csv"))

    if args.tune_on_full:
        thresholds = tune_thresholds(oof_probs, oof_labels)
        tuned_mf1 = compute_macro_f1(oof_probs, oof_labels, thresholds)
        print(f"[tune-on-full, NOT RECOMMENDED] Tuned OOF Macro F1: {tuned_mf1:.4f}")
    else:
        holdout_users = set(get_or_create_split())
        is_holdout = oof_meta["anon_user_id"].isin(holdout_users).values
        thresholds = tune_thresholds(oof_probs[~is_holdout], oof_labels[~is_holdout])
        honest_mf1 = compute_macro_f1(oof_probs[is_holdout], oof_labels[is_holdout], thresholds)
        full_mf1_for_reference = compute_macro_f1(oof_probs, oof_labels, thresholds)
        print(f"Thresholds tuned on 133 non-pseudo-leaderboard users only.")
        print(f"  Honest pseudo-leaderboard score (20 never-tuned-on users): {honest_mf1:.4f}  <- TRUST THIS")
        print(f"  Same thresholds applied to all 153 users (for reference)  : {full_mf1_for_reference:.4f}")

    with open(os.path.join(output_dir, "factor_thresholds_v2.json"), "w") as f:
        json.dump({"thresholds": {FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)},
                    "tune_on_full": args.tune_on_full}, f, indent=2)

    # ---- Leaderboard inference ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    add_risk_tokens(tokenizer)  # must exactly mirror train_v2.py

    sbert_lookup = load_sbert_lookup()
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    risk_df = pd.read_csv(args.risk_csv)
    risk_map = dict(zip(risk_df["row_id"], risk_df["risk_level"].str.lower()))
    missing = [rid for rid in lb_df["row_id"] if rid not in risk_map]
    if missing:
        raise ValueError(f"{len(missing)} leaderboard rows missing from --risk-csv, e.g. {missing[:5]}")
    lb_risk = [risk_map[rid] for rid in lb_df["row_id"]]

    lb_ds = FactorDatasetV2(lb_df, lb_risk, tokenizer, sbert_lookup, is_test=True)
    lb_loader = DataLoader(lb_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    model_paths = [os.path.join(output_dir, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS) if os.path.exists(os.path.join(output_dir, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} fold models")

    all_probs = []
    for p in model_paths:
        m = FactorClassifierV2(tokenizer=tokenizer).to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        fold_p = []
        with torch.no_grad():
            for batch in lb_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                sims = batch["sbert_sims"].to(device)
                ttype = batch.get("token_type_ids")
                if ttype is not None: ttype = ttype.to(device)
                logits, _ = m(ids, mask, sims, ttype)
                fold_p.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_probs.append(np.concatenate(fold_p))
        del m
        torch.cuda.empty_cache()

    avg = np.mean(all_probs, axis=0)
    np.save(os.path.join(output_dir, "leaderboard_probs.npy"), avg)

    preds = (avg >= thresholds).astype(int)
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"], "factors": str(factors) if factors else "[]"})
    out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "factor_predictions_v2.csv")
    out.to_csv(out_path, index=False)
    n = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"Saved -> {out_path}  (mean factors/post: {n.mean():.2f}, zero: {(n==0).sum()}/{len(out)})")


if __name__ == "__main__":
    main()
