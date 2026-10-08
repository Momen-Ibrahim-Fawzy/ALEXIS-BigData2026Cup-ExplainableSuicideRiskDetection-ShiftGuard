"""Subtask 1a: Generate leaderboard risk predictions (fold ensemble)."""

import os, json, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from config import *
from dataset import load_leaderboard, get_tokenizer, RiskDataset
from model import RiskClassifier


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    lb_df = load_leaderboard()
    ds = RiskDataset(lb_df, tokenizer, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    # Collect best models
    model_paths = []
    for fold in range(N_FOLDS):
        p = os.path.join(OUTPUT_DIR, f"fold_{fold}", "best.pt")
        if os.path.exists(p):
            model_paths.append(p)
    print(f"Ensembling {len(model_paths)} fold models")

    all_logits = []
    for path in model_paths:
        model = RiskClassifier().to(device)
        ckpt = torch.load(path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        fold_logits = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ttype = batch.get("token_type_ids")
                if ttype is not None:
                    ttype = ttype.to(device)
                logits = model(ids, mask, ttype)
                fold_logits.append(logits.cpu().numpy())
        all_logits.append(np.concatenate(fold_logits))

    avg = np.mean(all_logits, axis=0)
    preds = [ID2RISK[i] for i in avg.argmax(axis=1)]

    lb_df["risk_level"] = [p.capitalize() for p in preds]
    lb_df[["row_id", "risk_level"]].to_csv(
        os.path.join(OUTPUT_DIR, "risk_predictions.csv"), index=False)
    print(f"Saved to {OUTPUT_DIR}/risk_predictions.csv")
    print(lb_df["risk_level"].value_counts().to_string())
    return preds


if __name__ == "__main__":
    predict()
