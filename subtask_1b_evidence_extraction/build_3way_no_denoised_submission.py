"""Leaderboard predictions for the 1b ensemble composition vanilla+crf+
crf_denoised (true-OOF Phrase F1 0.8737, 2nd best of 25 subsets searched),
deliberately excluding 'denoised' -- the family whose solo real regression
(-0.0250) was the worst known Sub1b result. Also the top scorer on Sub1b's
newly-validated native population-shift check (weighted 0.8447 vs the
already-submitted 4-way composition's 0.8435).

Averages emission logits across the 3 families (5-fold-ensembled each,
matching this project's standard inference convention), decodes via the
averaged CRF transition matrix (2 of the 3 families -- crf, crf_denoised --
have a CRF layer; vanilla doesn't).

Usage:
    python build_3way_no_denoised_submission.py
"""

import os, json
import numpy as np
import pandas as pd
import torch

from config import *
from dataset import get_tokenizer
from model import EvidenceExtractor
from crf import CRF
from train import bio_to_spans

FAMILIES = {
    "vanilla":      ("outputs", False),
    "crf":          ("outputs_crf", True),
    "crf_denoised": ("outputs_crf_denoised", True),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_member(fam_dir, fold, has_crf):
    ckpt = torch.load(f"{fam_dir}/fold_{fold}/best.pt", weights_only=False, map_location=DEVICE)
    m = EvidenceExtractor().to(DEVICE).float()
    m.load_state_dict(ckpt["model"])
    m.eval()
    crf = None
    if has_crf:
        crf = CRF(NUM_BIO).to(DEVICE)
        crf.load_state_dict(ckpt["crf"])
        crf.eval()
    return m, crf


def average_crf_states(crf_state_dicts):
    keys = crf_state_dicts[0].keys()
    avg = {k: torch.stack([sd[k] for sd in crf_state_dicts]).mean(0) for k in keys}
    crf = CRF(NUM_BIO).to(DEVICE)
    crf.load_state_dict(avg)
    crf.eval()
    return crf


@torch.no_grad()
def main():
    tokenizer = get_tokenizer()
    lb_df = pd.read_excel(LEADERBOARD_PATH, engine="openpyxl")
    print(f"Leaderboard posts: {len(lb_df)}")

    # Pre-tokenize once, reuse across all 4 families x 5 folds
    encs = []
    for _, row in lb_df.iterrows():
        text = str(row["post"])
        enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True,
                        return_tensors="pt", return_offsets_mapping=True)
        encs.append({"ids": enc["input_ids"].to(DEVICE), "mask": enc["attention_mask"].to(DEVICE),
                     "offsets": enc["offset_mapping"].squeeze(0).tolist(), "text": text})

    all_family_logits = {fam: [] for fam in FAMILIES}  # fam -> list of (5-fold-avg) logits per row
    all_crf_states = []

    for fam, (fam_dir, has_crf) in FAMILIES.items():
        print(f"\n=== {fam} ({fam_dir}) ===")
        fold_logits = []  # 5 x (N, seq, NUM_BIO)
        fam_crf_states = []
        for fold in range(N_FOLDS):
            model, crf = load_member(fam_dir, fold, has_crf)
            row_logits = []
            for e in encs:
                logits = model(e["ids"], e["mask"]).float()
                row_logits.append(logits.cpu())
            fold_logits.append(torch.stack(row_logits))  # (N, 1, seq, NUM_BIO) after stack -> squeeze later
            if crf is not None:
                fam_crf_states.append({k: v.clone().cpu() for k, v in crf.state_dict().items()})
            del model
            if crf is not None:
                del crf
            torch.cuda.empty_cache()
            print(f"  fold {fold} done")
        avg_logits = torch.stack(fold_logits).mean(0)  # (N, 1, seq, NUM_BIO)
        all_family_logits[fam] = avg_logits
        if fam_crf_states:
            all_crf_states.extend(fam_crf_states)

    # Average across the 4 families
    stacked = torch.stack([all_family_logits[f] for f in FAMILIES])  # (4, N, 1, seq, NUM_BIO)
    ensemble_logits = stacked.mean(0)  # (N, 1, seq, NUM_BIO)

    crf_decoder = average_crf_states(all_crf_states) if all_crf_states else None

    pred_evidence = []
    for i, e in enumerate(encs):
        logits_i = ensemble_logits[i].to(DEVICE)  # (1, seq, NUM_BIO)
        mask_i = e["mask"].bool()  # the tokenizer's own attention_mask, not reconstructed
        if crf_decoder is not None:
            bio = crf_decoder.decode(logits_i, mask_i)[0]
        else:
            bio = logits_i.argmax(-1).cpu().numpy()[0]
        spans = bio_to_spans(bio, e["offsets"], e["text"])
        pred_evidence.append("; ".join(spans) if spans != ["none"] else "none")
        if (i + 1) % 50 == 0:
            print(f"  decoded {i+1}/{len(encs)}")

    out = pd.DataFrame({"row_id": lb_df["row_id"], "evidence": pred_evidence})
    out_path = os.path.join(os.path.dirname(__file__), "3way_no_denoised_1b_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved 1b predictions -> {out_path}")
    print(f"'none' rate: {(out['evidence']=='none').mean():.3f}")


if __name__ == "__main__":
    main()
