"""Generate leaderboard factor predictions from a trained GCN-gated Sub2
model (train_gcn.py). 5-fold ensemble average of sigmoid probs.

Usage:
    python predict_gcn.py --output-dir outputs_gcn_factor_verified
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config import *
from dataset import load_leaderboard, get_tokenizer, FactorDataset
from model_gcn import GCNFactorClassifier


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


def predict(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    lb_df = load_leaderboard()
    ds = FactorDataset(lb_df, tokenizer, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    adjacency = np.load(os.path.join(output_dir, "adjacency.npy"))
    factor_name_embeds = torch.load(os.path.join(output_dir, "factor_name_embeds.pt"), weights_only=False)

    model_paths = [os.path.join(output_dir, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS)
                   if os.path.exists(os.path.join(output_dir, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} fold models")

    all_probs = []
    for p in model_paths:
        m = GCNFactorClassifier(adjacency=adjacency, factor_name_embeds=factor_name_embeds).to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        fold_p = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ttype = batch.get("token_type_ids")
                if ttype is not None: ttype = ttype.to(device)
                logits = m(ids, mask, ttype)
                fold_p.append(torch.sigmoid(logits).cpu().numpy())
        all_probs.append(np.concatenate(fold_p))
        del m
        torch.cuda.empty_cache()

    avg = np.mean(all_probs, axis=0)
    np.save(os.path.join(output_dir, "leaderboard_probs.npy"), avg)

    oof_probs = np.load(os.path.join(output_dir, "oof_probs.npy"))
    oof_labels = np.load(os.path.join(output_dir, "oof_labels.npy"))
    thresholds = tune_thresholds(oof_probs, oof_labels)
    preds05 = (oof_probs >= 0.5).astype(int)
    tuned_preds = (oof_probs >= thresholds).astype(int)
    tuned_mf1 = np.mean([f1_score(oof_labels[:, i], tuned_preds[:, i], zero_division=0) for i in range(NUM_FACTORS)])
    print(f"Tuned OOF Macro F1: {tuned_mf1:.4f}")
    with open(os.path.join(output_dir, "factor_thresholds.json"), "w") as f:
        json.dump({"thresholds": {FACTOR_LIST[i]: float(thresholds[i]) for i in range(NUM_FACTORS)},
                    "tuned_macro_f1": float(tuned_mf1)}, f, indent=2)

    preds = (avg >= thresholds).astype(int)
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i, j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"], "factors": str(factors) if factors else "[]"})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir, "factor_predictions.csv"), index=False)
    out["n"] = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"Saved. Mean factors/post: {out['n'].mean():.1f}, zero: {(out['n']==0).sum()}/{len(out)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    predict(os.path.abspath(args.output_dir))
