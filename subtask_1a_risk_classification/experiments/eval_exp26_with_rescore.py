"""One-off evaluation: extract exp26's true-OOF logits, then run them through
the ALREADY-VALIDATED v1 sequence-rescorer (unmodified) to get the real,
decision-relevant number -- exactly mirroring what production does for
exp09, so the comparison to the 0.8078 baseline is apples-to-apples.

Usage:
    python eval_exp26_with_rescore.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, ID2RISK, RISK_LEVELS, BASE_DIR, DATA_DIR, RESULTS_DIR
from experiment_config import EXPERIMENTS
from sequence_rescore import SeqRescorer, UserSeqDataset, collate, build_users, D_MODEL, EPOCHS, PATIENCE, LR, CLASS_WEIGHTS, SEED

EXP_NAME = "exp26_deberta_large_contrastive_augment"
N_FOLDS = 5
PRODUCTION_WF1 = 0.8078
EXP09_BASELINE_WF1 = 0.8014


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EXPERIMENTS[EXP_NAME]
    exp_dir = os.path.join(RESULTS_DIR, EXP_NAME)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # ── Extract true-OOF logits (same discipline as extract_logits.py) ──────
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    oof_logits = np.zeros((len(df), 4), dtype=np.float64)
    filled = np.zeros(len(df), dtype=bool)

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        print(f"  [extract] fold {fold}: {len(va_i)} rows <- {ckpt_path}")
        model = Classifier(cfg["model_name"]).to(device).float()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        va_ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                             tokenizer, cfg["max_length"])
        va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"] * 2, shuffle=False, num_workers=2)
        logits_list = []
        with torch.no_grad():
            for batch in va_ld:
                ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                logits_list.append(model(ids, mask).float().cpu().numpy())
        oof_logits[va_i] = np.concatenate(logits_list, axis=0)
        filled[va_i] = True
        del model
        torch.cuda.empty_cache()
    assert filled.all()

    base_preds = oof_logits.argmax(axis=1)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    raw_wf1 = f1_score(true_labels, base_preds, average="weighted", zero_division=0)
    print(f"\nexp26 raw classifier, true stitched-OOF WF1: {raw_wf1:.4f}")
    print(f"Compare to: exp09 raw (true stitched-OOF) {EXP09_BASELINE_WF1:.4f}")

    # ── Run the UNMODIFIED v1 sequence-rescorer on top ───────────────────────
    train_meta = df[["row_id", "anon_user_id", "post_id", "risk_norm"]]
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
    all_rescored = np.zeros_like(oof_logits)

    for fold, (tr_i, va_i) in enumerate(gkf.split(train_meta, groups=groups)):
        torch.manual_seed(SEED + fold)
        tr_meta, va_meta = train_meta.iloc[tr_i], train_meta.iloc[va_i]
        tr_users = build_users(tr_meta, oof_logits, "risk_norm")
        va_users = build_users(va_meta, oof_logits, "risk_norm")
        tr_ld = DataLoader(UserSeqDataset(tr_users), batch_size=16, shuffle=True, collate_fn=lambda b: collate(b, True))
        va_ld = DataLoader(UserSeqDataset(va_users), batch_size=32, shuffle=False, collate_fn=lambda b: collate(b, True))

        model = SeqRescorer().to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_w, ignore_index=-100)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for logits_b, post_ids_b, pad_mask_b, labels_b, _ in tr_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b, labels_b = pad_mask_b.to(device), labels_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b)
                loss = criterion(out.reshape(-1, 4), labels_b.reshape(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for logits_b, post_ids_b, pad_mask_b, labels_b, _ in va_ld:
                    logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                    pad_mask_b = pad_mask_b.to(device)
                    out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                    mask_np = ~pad_mask_b.cpu().numpy()
                    preds_all.append(out[mask_np].argmax(axis=1))
                    labels_all.append(labels_b.numpy()[mask_np])
            preds_all, labels_all = np.concatenate(preds_all), np.concatenate(labels_all)
            wf1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    break
        print(f"  Fold {fold}: best val WF1={best_wf1:.4f}")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for logits_b, post_ids_b, pad_mask_b, labels_b, row_idx_b in va_ld:
                logits_b, post_ids_b = logits_b.to(device), post_ids_b.to(device)
                pad_mask_b = pad_mask_b.to(device)
                out = model(logits_b, post_ids_b, pad_mask_b).cpu().numpy()
                for bi, ridx in enumerate(row_idx_b):
                    all_rescored[ridx] = out[bi, :len(ridx)]
        del model
        torch.cuda.empty_cache()

    rescored_preds = all_rescored.argmax(axis=1)
    rescored_wf1 = f1_score(true_labels, rescored_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"exp26 raw:                      {raw_wf1:.4f}")
    print(f"exp26 + v1 sequence-rescoring:   {rescored_wf1:.4f}")
    print(f"exp09 raw (baseline):            {EXP09_BASELINE_WF1:.4f}")
    print(f"exp09 + v1 rescoring (PRODUCTION): {PRODUCTION_WF1:.4f}")
    print(f"exp26+rescore vs production:     {rescored_wf1 - PRODUCTION_WF1:+.4f}")
    print(f"{'='*60}")

    result = {"exp26_raw_wf1": float(raw_wf1), "exp26_rescored_wf1": float(rescored_wf1),
              "exp09_baseline": EXP09_BASELINE_WF1, "production_wf1": PRODUCTION_WF1}
    with open(os.path.join(exp_dir, "eval_with_rescore_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if rescored_wf1 > PRODUCTION_WF1:
        print("\n*** BEATS production -- worth a real submission. ***")
    else:
        print("\n*** Does NOT beat production. ***")


if __name__ == "__main__":
    main()
