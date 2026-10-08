"""
Error analysis for Sub1b: extract true OOF span predictions from the
"denoised" family (no CRF, the ingredient present in every top composition-
search result), compare against gold evidence spans, and save concrete
mismatches for manual reading -- mirroring the methodology that found the
farewell-note pattern for Sub1a.

Usage:
    python error_analysis.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from config import *
from dataset import load_train, get_tokenizer, EvidenceDataset
from model import EvidenceExtractor
from train import bio_to_spans, phrase_f1_single, normalize

FAMILY_DIR = "outputs_denoised"
N_FOLDS = 5


@torch.no_grad()
def predict_fold(fold_dir, va_df, tokenizer, device):
    ckpt = torch.load(os.path.join(fold_dir, "best.pt"), map_location=device, weights_only=False)
    model = EvidenceExtractor().to(device).float()
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = EvidenceDataset(va_df, tokenizer, is_test=True)
    ld = DataLoader(ds, batch_size=8, shuffle=False)

    all_spans = []
    idx = 0
    for batch in ld:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        offsets = batch["offset_mapping"]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, mask)
        bio_ids = logits.argmax(-1).cpu().numpy()
        for i in range(len(bio_ids)):
            text = str(va_df.iloc[idx]["post"])
            spans = bio_to_spans(bio_ids[i], offsets[i].tolist(), text)
            all_spans.append(spans)
            idx += 1

    del model
    torch.cuda.empty_cache()
    return all_spans


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    df = load_train()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    all_rows = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i].reset_index(drop=True)
        print(f"Fold {fold}: predicting {len(va_df)} posts...")
        pred_spans = predict_fold(os.path.join(FAMILY_DIR, f"fold_{fold}"), va_df, tokenizer, device)

        for i, row in va_df.iterrows():
            true_spans = row["evidence_list"] if row["evidence_list"] else ["none"]
            p, r, f1 = phrase_f1_single(pred_spans[i], true_spans)
            all_rows.append({
                "row_id": row["row_id"], "post": row["post"], "risk_norm": row["risk_norm"],
                "true_spans": "; ".join(true_spans), "pred_spans": "; ".join(pred_spans[i]),
                "precision": p, "recall": r, "f1": f1,
            })

    result_df = pd.DataFrame(all_rows)
    mean_f1 = result_df["f1"].mean()
    print(f"\nTrue OOF mean per-post Phrase F1 (denoised family): {mean_f1:.4f}")

    result_df.to_csv("/tmp/1b_denoised_oof_errors.csv", index=False)
    print("Saved -> /tmp/1b_denoised_oof_errors.csv")

    exact_none_both = ((result_df.true_spans == "none") & (result_df.pred_spans == "none")).sum()
    missed_evidence = ((result_df.true_spans != "none") & (result_df.pred_spans == "none")).sum()
    false_evidence = ((result_df.true_spans == "none") & (result_df.pred_spans != "none")).sum()
    both_have_evidence = ((result_df.true_spans != "none") & (result_df.pred_spans != "none")).sum()
    print(f"\nBoth 'none' (correct no-evidence): {exact_none_both}")
    print(f"Missed evidence (true has spans, pred says none): {missed_evidence}")
    print(f"False evidence (true is none, pred found spans): {false_evidence}")
    print(f"Both have evidence (need span-overlap check): {both_have_evidence}")


if __name__ == "__main__":
    main()
