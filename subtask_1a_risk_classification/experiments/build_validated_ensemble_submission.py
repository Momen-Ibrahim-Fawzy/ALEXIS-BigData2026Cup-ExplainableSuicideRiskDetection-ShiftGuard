"""
Builds leaderboard-set predictions for the honest-check-validated ensemble
(exp09 + FGM + self-training + FreeLB + CORAL + R-Drop), selected via greedy
forward selection on the 133-user tune pool only and confirmed +0.0086 WF1 on
the never-touched 20-user holdout (see ensemble_negatives_check.py).

For each of the 6 techniques: 5-fold-ensemble average on the leaderboard set
(same convention as every other production inference in this project), then
uniform-average the 6 technique-level results together.

Usage:
    python build_validated_ensemble_submission.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, ID2RISK, DATA_DIR, RESULTS_DIR
from coral_model import CoralDualHeadClassifier

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
N_FOLDS = 5
BATCH_SIZE = 32

SELECTED = [
    ("exp32_deberta_large_fgm", "plain"),
    ("exp33_deberta_large_selftrain", "plain"),
    ("exp36_deberta_large_freelb", "plain"),
    ("exp38_deberta_coral_weighted", "coral"),
    ("exp39_deberta_large_rdrop", "plain"),
]


@torch.no_grad()
def predict_5fold_avg(exp_dir, arch, lb_ds, device):
    lb_ld = DataLoader(lb_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(RESULTS_DIR, exp_dir, f"best_fold{fold}.pt")
        if arch == "coral":
            model = CoralDualHeadClassifier(MODEL_NAME).to(device).float()
        else:
            model = Classifier(MODEL_NAME).to(device).float()
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        probs_this_fold = []
        for batch in lb_ld:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if arch == "coral":
                    sl, cl = model(ids, mask)
                    probs = model.combined_probs(sl.float(), cl.float())
                else:
                    logits = model(ids, mask)
                    probs = torch.softmax(logits.float(), dim=-1)
            probs_this_fold.append(probs.cpu().numpy())
        fold_probs.append(np.concatenate(probs_this_fold, axis=0))
        print(f"  {exp_dir} fold {fold} done")
        del model
        torch.cuda.empty_cache()

    return np.mean(fold_probs, axis=0)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    print(f"Leaderboard posts: {len(lb_df)}")
    lb_ds = TextDataset(lb_df["post"].tolist(), None, tokenizer, MAX_LENGTH)

    # exp09's leaderboard logits already exist from the sequence-rescoring extraction
    exp09_logits = np.load(os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high",
                                          "sequence_data", "leaderboard_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)

    all_probs = {"exp09": exp09_probs}
    for exp_dir, arch in SELECTED:
        print(f"\nPredicting leaderboard set: {exp_dir} ({arch})")
        all_probs[exp_dir] = predict_5fold_avg(exp_dir, arch, lb_ds, device)

    ensemble_probs = np.mean(list(all_probs.values()), axis=0)
    np.save(os.path.join(RESULTS_DIR, "validated_ensemble_leaderboard_probs.npy"), ensemble_probs)

    pred_ids = ensemble_probs.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]
    print("\nPrediction distribution:")
    print(pd.Series(pred_labels).value_counts().to_string())

    out = pd.DataFrame({"row_id": lb_df["row_id"], "risk_level": pred_labels})
    out_path = os.path.join(RESULTS_DIR, "validated_ensemble_1a_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved 1a predictions -> {out_path}")
    print("Members used:", list(all_probs.keys()))


if __name__ == "__main__":
    main()
