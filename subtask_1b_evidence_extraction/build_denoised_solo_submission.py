"""Leaderboard predictions for 'denoised' ALONE -- no ensembling, no search.

Subtask 1b's only real submission so far (the 4-way composition, found via
exhaustive search over every 2/3/4-way subset of 5 families) regressed
-0.0196 Phrase F1 for real. Sub1a's own real-submission history shows
search-selected candidates have the largest local-to-real gaps of anything
tried (v42/v43, gaps of 0.051/0.074 vs single-model changes' 0.028-0.040).
This tests whether the SAME pattern holds for 1b: a single, non-searched
family, not chosen as the best of many options.

'denoised' is the single strongest non-ensembled family locally (true-OOF
Phrase F1 0.8460, vs the composition's search-selected 0.8724) and the only
one used in every combination that was tried during the search.

Usage:
    python build_denoised_solo_submission.py
"""

import os
import numpy as np
import pandas as pd
import torch

from config import *
from dataset import get_tokenizer
from model import EvidenceExtractor
from train import bio_to_spans

FAM_DIR = "outputs_denoised"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_member(fold):
    ckpt = torch.load(f"{FAM_DIR}/fold_{fold}/best.pt", weights_only=False, map_location=DEVICE)
    m = EvidenceExtractor().to(DEVICE).float()
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


@torch.no_grad()
def main():
    tokenizer = get_tokenizer()
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    print(f"Leaderboard posts: {len(lb_df)}")

    encs = []
    for _, row in lb_df.iterrows():
        text = str(row["post"])
        enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True,
                        return_tensors="pt", return_offsets_mapping=True)
        encs.append({"ids": enc["input_ids"].to(DEVICE), "mask": enc["attention_mask"].to(DEVICE),
                     "offsets": enc["offset_mapping"].squeeze(0).tolist(), "text": text})

    fold_logits = []
    for fold in range(N_FOLDS):
        model = load_member(fold)
        row_logits = []
        for e in encs:
            logits = model(e["ids"], e["mask"]).float()
            row_logits.append(logits.cpu())
        fold_logits.append(torch.stack(row_logits))
        del model
        torch.cuda.empty_cache()
        print(f"  fold {fold} done")

    avg_logits = torch.stack(fold_logits).mean(0)  # (N, 1, seq, NUM_BIO)

    pred_evidence = []
    for i, e in enumerate(encs):
        logits_i = avg_logits[i]
        bio = logits_i.argmax(-1).cpu().numpy()[0]
        spans = bio_to_spans(bio, e["offsets"], e["text"])
        pred_evidence.append("; ".join(spans) if spans != ["none"] else "none")
        if (i + 1) % 50 == 0:
            print(f"  decoded {i+1}/{len(encs)}")

    out = pd.DataFrame({"row_id": lb_df["row_id"], "evidence": pred_evidence})
    out_path = os.path.join(os.path.dirname(__file__), "denoised_solo_1b_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved 1b predictions -> {out_path}")
    print(f"'none' rate: {(out['evidence']=='none').mean():.3f}")


if __name__ == "__main__":
    main()
