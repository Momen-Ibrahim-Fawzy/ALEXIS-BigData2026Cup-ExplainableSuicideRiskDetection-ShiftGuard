"""
Self-training step 1: pseudo-label the 378 unlabeled leaderboard posts using
exp09's confirmed-real-leaderboard 5-fold ensemble (the production Sub1a
component), then write a JSON in the same schema as the other oversample_*
sources so it can be dropped into trainer.py's load_data() ov_files map.

Never tried in this project. Directly transfers the single most reliable
compounding technique from a sibling project's Subtask 1 (DialectSentEval),
where 5 successive self-training rounds each improved real leaderboard F1
with no reversal (0.8204 -> 0.8307 -> 0.8459 -> 0.8477 -> ...).

Usage:
    python pseudo_label_leaderboard.py
"""

import os, sys, json
import torch
import pandas as pd
from transformers import AutoTokenizer

from trainer import Classifier, RISK_LEVELS, ID2RISK, DATA_DIR

EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                        "exp09_deberta_large_lr_high")
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
N_FOLDS = 5


@torch.no_grad()
def predict_probs(texts, device):
    """5-fold ensemble: average softmax probabilities across exp09's fold checkpoints."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_fold_probs = []

    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(EXP_DIR, f"best_fold{fold}.pt")
        model = Classifier(MODEL_NAME).to(device).float()
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        fold_probs = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, max_length=MAX_LENGTH, padding="max_length",
                             truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            fold_probs.append(probs)
        all_fold_probs.append(torch.cat(fold_probs, dim=0))
        print(f"  Fold {fold} done ({len(texts)} posts)")

        del model
        torch.cuda.empty_cache()

    # Average across the 5 folds -- same ensembling exp09 uses at real inference time
    avg_probs = torch.stack(all_fold_probs, dim=0).mean(dim=0)
    return avg_probs, torch.stack(all_fold_probs, dim=0)  # (avg, per_fold)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lb_path = os.path.join(DATA_DIR, "leaderboard.xlsx")
    df = pd.read_excel(lb_path, engine="openpyxl")
    print(f"Leaderboard posts: {len(df)}")

    texts = df["post"].astype(str).tolist()
    avg_probs, per_fold_probs = predict_probs(texts, device)

    max_probs, pred_ids = avg_probs.max(dim=-1), avg_probs.argmax(dim=-1)
    confidences = max_probs.values.tolist()
    preds = [ID2RISK[i.item()] for i in pred_ids]

    # Committee agreement: how many of the 5 fold models agree with the ensemble argmax
    per_fold_argmax = per_fold_probs.argmax(dim=-1)  # (5, N)
    agreement = (per_fold_argmax == pred_ids.unsqueeze(0)).float().mean(dim=0).tolist()

    out_rows = []
    for i, row in df.iterrows():
        out_rows.append({
            "row_id": row["row_id"],
            "text": row["post"],
            "risk_level": preds[i],
            "confidence": round(confidences[i], 4),
            "committee_agreement": round(agreement[i], 4),
        })

    out_path = os.path.join(DATA_DIR, "pseudo_leaderboard_full.json")
    with open(out_path, "w") as f:
        json.dump(out_rows, f, indent=2)
    print(f"\nWrote {len(out_rows)} pseudo-labeled rows to {out_path}")

    # Report confidence distribution to inform the threshold choice
    import numpy as np
    conf_arr = np.array(confidences)
    agree_arr = np.array(agreement)
    print("\nConfidence distribution (max avg-softmax prob):")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  p{int(q*100)}: {np.quantile(conf_arr, q):.4f}")
    print(f"  mean: {conf_arr.mean():.4f}")

    print("\nCommittee agreement (fraction of 5 folds matching ensemble argmax):")
    for thresh in [1.0, 0.8, 0.6]:
        n = (agree_arr >= thresh).sum()
        print(f"  >= {thresh}: {n}/{len(agree_arr)} ({100*n/len(agree_arr):.1f}%)")

    print("\nPredicted risk-level distribution:")
    print(pd.Series(preds).value_counts())

    for thresh in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
        n = (conf_arr >= thresh).sum()
        print(f"  confidence >= {thresh}: {n}/{len(conf_arr)} ({100*n/len(conf_arr):.1f}%)")


if __name__ == "__main__":
    main()
