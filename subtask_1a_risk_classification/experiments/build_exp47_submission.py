"""Leaderboard predictions for exp47 (RSD-15K two-stage transfer learning)
standalone -- no blending with exp09. Built despite the local checks not
clearing: true-OOF is the best margin of the session (+0.0174), but the
importance-weighted check only shrinks (doesn't flip, same bucket as 3
confirmed-real regressions) and exp47 actually does WORSE than exp09 on a
genuinely held-out RSD-15K slice it never trained on. Built anyway, at
explicit request, to get the real ground truth on whether external-data
transfer learning helps -- a genuinely new category never tested for real
this session, unlike every prior recombination-of-existing-checkpoints
candidate.

Usage:
    python build_exp47_submission.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, ID2RISK, DATA_DIR, RESULTS_DIR

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
N_FOLDS = 5
BATCH_SIZE = 16
EXP_DIR = os.path.join(RESULTS_DIR, "exp47_rsd15k_two_stage")


@torch.no_grad()
def predict_5fold_avg(lb_ds, device):
    lb_ld = DataLoader(lb_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        model = Classifier(MODEL_NAME).to(device).float()
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        probs_this_fold = []
        for batch in lb_ld:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
                probs = torch.softmax(logits.float(), dim=-1)
            probs_this_fold.append(probs.cpu().numpy())
        fold_probs.append(np.concatenate(probs_this_fold, axis=0))
        print(f"  fold {fold} done", flush=True)
        del model
        torch.cuda.empty_cache()

    return np.mean(fold_probs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    print(f"Leaderboard posts: {len(lb_df)}")
    lb_ds = TextDataset(lb_df["post"].tolist(), None, tokenizer, MAX_LENGTH)

    probs = predict_5fold_avg(lb_ds, device)
    np.save(os.path.join(RESULTS_DIR, "exp47_leaderboard_probs.npy"), probs)

    pred_ids = probs.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    print("\nPrediction distribution:")
    print(pd.Series(pred_labels).value_counts().to_string())

    exp09_logits = np.load(os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high",
                                          "sequence_data", "leaderboard_logits.npy"))
    exp09_labels = [ID2RISK[i].capitalize() for i in exp09_logits.argmax(axis=1)]
    n_changed = sum(a != b for a, b in zip(pred_labels, exp09_labels))
    print(f"\nRows where exp47 label differs from exp09-alone: {n_changed}/{len(lb_df)}")

    out = pd.DataFrame({"row_id": lb_df["row_id"], "risk_level": pred_labels})
    out_path = os.path.join(RESULTS_DIR, "exp47_1a_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved 1a predictions -> {out_path}")


if __name__ == "__main__":
    main()
