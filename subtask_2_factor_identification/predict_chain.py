"""Generate leaderboard factor predictions from a trained chain model
(train_chain.py). 5-fold ensemble average of sigmoid probs (self-conditioned
chaining, no teacher labels -- matches eval_epoch's behavior in train_chain.py
since true labels don't exist for the leaderboard).

Usage:
    python predict_chain.py --output-dir outputs_chain_factor_verified
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import *
from dataset import load_leaderboard, get_tokenizer, FactorDataset
from model_chain import ChainedFactorClassifier
from train_chain import compute_macro_f1


def tune_thresholds(probs, labels):
    candidates = np.arange(0.05, 0.95, 0.025)
    thresholds = np.full(NUM_FACTORS, 0.5)
    for i in range(NUM_FACTORS):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            preds = (probs[:, i] >= t).astype(int)
            from sklearn.metrics import f1_score
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

    with open(os.path.join(output_dir, "chain_order.json")) as f:
        order = json.load(f)["order"]

    model_paths = [os.path.join(output_dir, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS)
                   if os.path.exists(os.path.join(output_dir, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} fold models")

    all_probs = []
    for p in model_paths:
        m = ChainedFactorClassifier(order=order).to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        fold_p = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ttype = batch.get("token_type_ids")
                if ttype is not None: ttype = ttype.to(device)
                logits = m(ids, mask, ttype, teacher_labels=None)
                fold_p.append(torch.sigmoid(logits).cpu().numpy())
        all_probs.append(np.concatenate(fold_p))
        del m
        torch.cuda.empty_cache()

    avg = np.mean(all_probs, axis=0)
    np.save(os.path.join(output_dir, "leaderboard_probs.npy"), avg)

    oof_probs = np.load(os.path.join(output_dir, "oof_probs.npy"))
    oof_labels = np.load(os.path.join(output_dir, "oof_labels.npy"))
    thresholds = tune_thresholds(oof_probs, oof_labels)
    tuned_mf1, _ = compute_macro_f1(oof_probs, oof_labels, thresholds)
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
