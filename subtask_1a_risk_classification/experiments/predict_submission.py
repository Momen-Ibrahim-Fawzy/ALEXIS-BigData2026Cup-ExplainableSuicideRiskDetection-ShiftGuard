"""
Generate leaderboard submission from a trained experiment's 5-fold ensemble.

Usage:
    # Best single experiment (default)
    python predict_submission.py

    # Specify one experiment
    python predict_submission.py exp09_deberta_large_lr_high

    # Cross-experiment ensemble (average all listed experiments)
    python predict_submission.py exp09_deberta_large_lr_high exp01_deberta_large exp14_mental_roberta_large

    # Apply OOF-tuned per-class bias calibration (see tune_thresholds_oof.py)
    python predict_submission.py exp09_deberta_large_lr_high --calibrate
"""

import os, sys, json, argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from trainer import (
    Classifier, TextDataset,
    RISK2ID, ID2RISK, RISK_LEVELS, SEED,
    BASE_DIR, DATA_DIR, RESULTS_DIR,
)
from experiment_config import EXPERIMENTS

LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.xlsx")
SUBMISSIONS_DIR  = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def get_logits(exp_name, cfg, lb_df, device):
    """Run all available fold models for one experiment and return averaged logits."""
    exp_dir   = os.path.join(RESULTS_DIR, exp_name)
    tokenizer_name = cfg["model_name"]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = TextDataset(
        lb_df["post"].tolist(),
        [0] * len(lb_df),          # dummy labels — not used
        tokenizer,
        cfg["max_length"],
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    fold_logits = []
    for fold in range(5):
        # Prefer corrected checkpoint for fold 0 if it exists
        corrected = os.path.join(exp_dir, "best_fold0_corrected.pt")
        standard  = os.path.join(exp_dir, f"best_fold{fold}.pt")

        if fold == 0 and os.path.exists(corrected):
            ckpt_path = corrected
            tag = "best_fold0_corrected.pt"
        elif os.path.exists(standard):
            ckpt_path = standard
            tag = f"best_fold{fold}.pt"
        else:
            print(f"    [fold {fold}] checkpoint not found — skipping")
            continue

        print(f"  [{exp_name}] fold {fold} → {tag}")
        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        logits_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"    fold {fold}", leave=False,
                              dynamic_ncols=True):
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                out  = model(ids, mask)
                logits_list.append(out.cpu().numpy())

        fold_logits.append(np.concatenate(logits_list, axis=0))
        del model
        torch.cuda.empty_cache()

    if not fold_logits:
        raise RuntimeError(f"No fold checkpoints found for {exp_name}")

    return np.mean(fold_logits, axis=0)   # shape (N, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*",
                        default=["exp09_deberta_large_lr_high"],
                        help="Experiment names to ensemble")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply OOF-tuned per-class logit bias "
                             "(from tune_thresholds_oof.py) before argmax. "
                             "Only meaningful for a single-experiment run.")
    args = parser.parse_args()

    exp_names = args.experiments
    for name in exp_names:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiments to ensemble: {exp_names}")

    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    print(f"Leaderboard: {len(lb_df)} posts")

    all_logits = []
    for name in exp_names:
        cfg = EXPERIMENTS[name]
        logits = get_logits(name, cfg, lb_df, device)
        all_logits.append(logits)

    avg_logits = np.mean(all_logits, axis=0)            # (N, 4)

    tag_suffix = ""
    if args.calibrate:
        if len(exp_names) != 1:
            print("ERROR: --calibrate only supported for a single experiment.")
            sys.exit(1)
        calib_path = os.path.join(RESULTS_DIR, exp_names[0], "threshold_calibration.json")
        if not os.path.exists(calib_path):
            print(f"ERROR: {calib_path} not found. Run tune_thresholds_oof.py first.")
            sys.exit(1)
        with open(calib_path) as f:
            calib = json.load(f)
        bias = np.array(calib["bias_vector"])
        print(f"\nApplying OOF-tuned bias: {bias.tolist()} "
              f"(OOF WF1 {calib['baseline_wf1']:.4f} -> {calib['tuned_wf1']:.4f})")
        avg_logits = avg_logits + bias
        tag_suffix = "_calibrated"

    pred_ids   = avg_logits.argmax(axis=1)
    pred_labels = [ID2RISK[i].capitalize() for i in pred_ids]

    # Build submission
    submission = pd.DataFrame({
        "row_id":     lb_df["row_id"],
        "risk_level": pred_labels,
    })

    # Distribution check
    print("\nPrediction distribution:")
    print(submission["risk_level"].value_counts().to_string())

    # Save
    tag = "_".join(exp_names) if len(exp_names) <= 3 else f"{len(exp_names)}exp_ensemble"
    out_path = os.path.join(SUBMISSIONS_DIR, f"submission_{tag}{tag_suffix}.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
