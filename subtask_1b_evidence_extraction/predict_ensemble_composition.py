"""Sub1b: generate leaderboard predictions from the winning composition found
by search_ensemble_composition.py -- vanilla + denoised + crf_denoised
(true OOF PhraseF1 0.7583, +0.0159 over production; honest pseudo-leaderboard
check even stronger at 0.8286).

Mechanism: average emission logits across all 15 fold models (5 vanilla +
5 denoised + 5 crf_denoised), decode via the crf_denoised family's own
5-fold-averaged CRF transition matrix (the only member with a CRF layer),
same "must match the validation script exactly" discipline as
predict_crf_vanilla_ensemble.py.

Usage:
    python predict_ensemble_composition.py --risk-csv <path to Sub1a risk CSV>
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from config import *
from dataset import load_leaderboard, get_tokenizer
from model import EvidenceExtractor
from crf import CRF
from train import bio_to_spans

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MEMBERS = [
    ("outputs", False),
    ("outputs_denoised", False),
    ("outputs_crf_denoised", True),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--risk-csv", default=None)
    p.add_argument("--output-dir", default="outputs_ensemble_composition")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer()
    lb_df = load_leaderboard()

    risk_map = {}
    if args.risk_csv and os.path.exists(args.risk_csv):
        rdf = pd.read_csv(args.risk_csv)
        risk_map = dict(zip(rdf["row_id"], rdf["risk_level"].str.lower()))
        print(f"Using risk predictions from {args.risk_csv}")

    # Memory-conscious: load one fold model at a time (not all 15 simultaneously --
    # 15x DeBERTa-v3-large would need ~25GB+, more than reliably free on a shared GPU),
    # accumulate a running emission-logit sum across all leaderboard posts.
    n_models = 0
    crf_states = []
    sum_logits = None  # (N, seq_len, NUM_BIO), accumulated on CPU
    all_enc = []
    for idx in range(len(lb_df)):
        text = str(lb_df.iloc[idx]["post"])
        enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                         truncation=True, return_tensors="pt", return_offsets_mapping=True)
        all_enc.append(enc)

    for fam_dir, has_crf in MEMBERS:
        for fold in range(N_FOLDS):
            ckpt = torch.load(f"{fam_dir}/fold_{fold}/best.pt", weights_only=False, map_location=DEVICE)
            m = EvidenceExtractor().to(DEVICE).float()
            m.load_state_dict(ckpt["model"])
            m.eval()
            if has_crf:
                crf_states.append(ckpt["crf"])

            with torch.no_grad():
                for idx, enc in enumerate(all_enc):
                    ids = enc["input_ids"].to(DEVICE)
                    mask = enc["attention_mask"].to(DEVICE)
                    ttype = enc.get("token_type_ids")
                    if ttype is not None:
                        ttype = ttype.to(DEVICE)
                    logit = m(ids, mask, ttype).float().cpu()
                    if sum_logits is None:
                        sum_logits = torch.zeros(len(all_enc), *logit.shape[1:])
                    sum_logits[idx] += logit.squeeze(0)

            n_models += 1
            print(f"  processed {fam_dir}/fold_{fold} ({n_models}/{sum(N_FOLDS for _ in MEMBERS)} models)")
            del m
            torch.cuda.empty_cache()

    avg_logits_all = (sum_logits / n_models).to(DEVICE)
    print(f"Averaged {n_models} emission models, {len(crf_states)} CRF transition sets")

    ensembled_crf = CRF(NUM_BIO).to(DEVICE)
    with torch.no_grad():
        avg_state = {k: torch.stack([sd[k] for sd in crf_states]).mean(0) for k in crf_states[0]}
        ensembled_crf.load_state_dict(avg_state)
    ensembled_crf.eval()

    rows = []
    for idx in range(len(lb_df)):
        row = lb_df.iloc[idx]
        rid = row["row_id"]
        text = str(row["post"])

        if risk_map.get(rid, "") == "indicator":
            rows.append({"row_id": rid, "evidence": "none"})
            continue

        enc = all_enc[idx]
        mask = enc["attention_mask"].to(DEVICE)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        with torch.no_grad():
            bio_preds = ensembled_crf.decode(avg_logits_all[idx:idx+1], mask.bool())[0]

        spans = bio_to_spans(bio_preds, offsets, text)
        ev_str = "; ".join(s for s in spans if s.lower() != "none") or "none"
        rows.append({"row_id": rid, "evidence": ev_str})

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(lb_df)}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, "evidence_predictions.csv")
    out.to_csv(out_path, index=False)
    has_ev = (out["evidence"] != "none").sum()
    print(f"Saved -> {out_path}  (posts with evidence: {has_ev}/{len(out)})")


if __name__ == "__main__":
    main()
