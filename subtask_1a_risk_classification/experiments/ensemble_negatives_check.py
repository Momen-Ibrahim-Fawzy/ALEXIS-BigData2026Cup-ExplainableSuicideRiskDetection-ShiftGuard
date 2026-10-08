"""
Checks whether ensembling today's negative-result experiments together beats
exp09 alone or the confirmed production combo (exp09 + v1 rescorer, true OOF
0.8078). Inference only -- reuses each experiment's already-trained fold
checkpoints, no new training.

All experiments below use the SAME GroupKFold(n_splits=5) split on the SAME
train.xlsx (deterministic, no randomness in GroupKFold itself), so fold
assignments are identical across experiments -- required for valid OOF
ensembling.

Usage:
    python ensemble_negatives_check.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, ID2RISK, RISK_LEVELS, DATA_DIR, RESULTS_DIR
from coral_model import CoralDualHeadClassifier

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
N_FOLDS = 5
BATCH_SIZE = 32

# (experiment_dir_name, architecture) -- "plain" = Classifier + softmax,
# "coral" = CoralDualHeadClassifier + combined_probs()
EXPERIMENTS = [
    ("exp32_deberta_large_fgm", "plain"),
    ("exp33_deberta_large_selftrain", "plain"),
    ("exp34_deberta_large_supcon", "plain"),
    ("exp36_deberta_large_freelb", "plain"),
    ("exp37_deberta_large_lookahead_ema", "plain"),
    ("exp38_deberta_coral_weighted", "coral"),
    ("exp39_deberta_large_rdrop", "plain"),
]


@torch.no_grad()
def get_fold_probs(exp_dir, arch, fold, va_ds, device, tokenizer):
    ckpt_path = os.path.join(RESULTS_DIR, exp_dir, f"best_fold{fold}.pt")
    if not os.path.exists(ckpt_path):
        print(f"    MISSING checkpoint: {ckpt_path}")
        return None

    if arch == "coral":
        model = CoralDualHeadClassifier(MODEL_NAME).to(device).float()
    else:
        model = Classifier(MODEL_NAME).to(device).float()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_probs = []
    for batch in va_ld:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if arch == "coral":
                standard_logits, coral_logits = model(ids, mask)
                probs = model.combined_probs(standard_logits.float(), coral_logits.float())
            else:
                logits = model(ids, mask)
                probs = torch.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_probs, axis=0)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    # exp09 baseline OOF logits (already extracted, convert to probs)
    exp09_logits = np.load(os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high",
                                          "sequence_data", "train_oof_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)
    exp09_wf1 = f1_score(true_labels, exp09_probs.argmax(1), average="weighted", zero_division=0)
    print(f"exp09 alone (reference): WF1 = {exp09_wf1:.4f}")

    all_oof_probs = {}  # exp_dir -> (1635, 4) array, filled fold by fold

    for exp_dir, arch in EXPERIMENTS:
        print(f"\nExtracting OOF probs: {exp_dir} ({arch})")
        oof = np.zeros((len(df), 4))
        for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
            va_df = df.iloc[va_i]
            va_ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                                 tokenizer, MAX_LENGTH)
            probs = get_fold_probs(exp_dir, arch, fold, va_ds, device, tokenizer)
            if probs is None:
                oof = None
                break
            oof[va_i] = probs
            print(f"  fold {fold} done")
        if oof is not None:
            wf1 = f1_score(true_labels, oof.argmax(1), average="weighted", zero_division=0)
            print(f"  {exp_dir} standalone true OOF WF1 (sanity check): {wf1:.4f}")
            all_oof_probs[exp_dir] = oof

    # Save individual results
    np.savez(os.path.join(RESULTS_DIR, "ensemble_negatives_oof_probs.npz"),
              exp09=exp09_probs, **all_oof_probs)

    # Try: exp09 + each successfully-extracted experiment, uniformly averaged (all together)
    members = {"exp09": exp09_probs, **all_oof_probs}
    print(f"\n{'='*60}\nEnsemble results (uniform average)\n{'='*60}")

    all_avg = np.mean(list(members.values()), axis=0)
    all_wf1 = f1_score(true_labels, all_avg.argmax(1), average="weighted", zero_division=0)
    print(f"ALL {len(members)} members (exp09 + all 7 negatives): WF1 = {all_wf1:.4f}")

    # Try: exp09 alone + just the single closest-to-baseline negative (CORAL)
    if "exp38_deberta_coral_weighted" in all_oof_probs:
        pair = np.mean([exp09_probs, all_oof_probs["exp38_deberta_coral_weighted"]], axis=0)
        pair_wf1 = f1_score(true_labels, pair.argmax(1), average="weighted", zero_division=0)
        print(f"exp09 + CORAL only: WF1 = {pair_wf1:.4f}")

    # Try: exp09 + all NEGATIVES excluding the two worst (SupCon, Lookahead+EMA)
    trimmed = {k: v for k, v in members.items()
               if k not in ("exp34_deberta_large_supcon", "exp37_deberta_large_lookahead_ema")}
    trimmed_avg = np.mean(list(trimmed.values()), axis=0)
    trimmed_wf1 = f1_score(true_labels, trimmed_avg.argmax(1), average="weighted", zero_division=0)
    print(f"exp09 + 5 negatives (excl. 2 worst): WF1 = {trimmed_wf1:.4f}")

    print(f"\nReference points:")
    print(f"  exp09 alone: {exp09_wf1:.4f}")
    print(f"  Confirmed production (exp09 + v1 rescorer): 0.8078")

    results = {
        "exp09_alone": float(exp09_wf1),
        "all_members_ensemble": float(all_wf1),
        "trimmed_ensemble": float(trimmed_wf1),
        "confirmed_production_reference": 0.8078,
        "members_used": list(members.keys()),
    }
    with open(os.path.join(RESULTS_DIR, "ensemble_negatives_check_result.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
