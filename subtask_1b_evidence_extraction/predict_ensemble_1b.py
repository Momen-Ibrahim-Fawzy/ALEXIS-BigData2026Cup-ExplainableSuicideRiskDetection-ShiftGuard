"""
General N-way 1b evidence ensemble: average emission logits across any set of
output directories (each containing fold_0/best.pt … fold_4/best.pt checkpoints).

Usage:
  # Default: vanilla + CRF (same as v21, 10 models)
  python predict_ensemble_1b.py --risk-csv path/to/risk.csv

  # 4-way seed ensemble (20 models)
  python predict_ensemble_1b.py --risk-csv risk.csv \\
      --dirs outputs outputs_crf outputs_seed100 outputs_seed200 \\
      --out evidence_predictions_4way.csv
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import MAX_LENGTH, N_FOLDS
from dataset import load_leaderboard, get_tokenizer
from model import EvidenceExtractor
from train import bio_to_spans

DEFAULT_DIRS = ["outputs", "outputs_crf"]


def get_ckpt_paths(dirs):
    paths = []
    for d in dirs:
        full_d = d if os.path.isabs(d) else os.path.join(BASE_DIR, d)
        for fold in range(N_FOLDS):
            ckpt_path = os.path.join(full_d, f"fold_{fold}", "best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  WARNING: {ckpt_path} not found — skipping")
                continue
            paths.append(ckpt_path)
    print(f"Ensemble: {len(paths)} checkpoints ({len(dirs)} dir(s) × up to {N_FOLDS} folds)")
    return paths


def predict(dirs, risk_csv=None, out_path=None):
    if out_path is None:
        n = len(dirs)
        out_path = os.path.join(BASE_DIR, f"evidence_predictions_{n}way.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer()
    lb_df = load_leaderboard()
    n_posts = len(lb_df)
    print(f"Leaderboard: {n_posts} posts")

    risk_map = {}
    if risk_csv and os.path.exists(risk_csv):
        rdf = pd.read_csv(risk_csv)
        risk_map = dict(zip(rdf["row_id"], rdf["risk_level"].str.lower()))
        print(f"Risk CSV: {len(risk_map)} indicator-suppressed posts")

    ckpt_paths = get_ckpt_paths(dirs)
    if not ckpt_paths:
        raise RuntimeError("No checkpoints found in any of the specified directories.")

    # Pre-encode all posts once to avoid re-tokenising per model
    indicator_mask = []
    encodings = []
    offsets_list = []
    for idx in range(n_posts):
        row = lb_df.iloc[idx]
        rid = row["row_id"]
        if risk_map.get(rid, "") == "indicator":
            indicator_mask.append(True)
            encodings.append(None)
            offsets_list.append(None)
        else:
            indicator_mask.append(False)
            enc = tokenizer(str(row["post"]), max_length=MAX_LENGTH,
                            padding="max_length", truncation=True,
                            return_tensors="pt", return_offsets_mapping=True)
            encodings.append(enc)
            offsets_list.append(enc["offset_mapping"].squeeze(0).tolist())

    # Accumulate logit sums in CPU numpy — shape (n_posts, MAX_LENGTH, n_bio_classes)
    # We accumulate lazily: only allocate on first non-indicator post with a real model pass
    bio_sums = [None] * n_posts

    # Load one model at a time, run all posts, free GPU memory before next model
    for ckpt_path in ckpt_paths:
        print(f"  Running {os.path.relpath(ckpt_path, BASE_DIR)} ...")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        m = EvidenceExtractor().to(device).float()
        m.load_state_dict(ckpt["model"])
        m.eval()
        del ckpt

        with torch.no_grad():
            for idx in range(n_posts):
                if indicator_mask[idx]:
                    continue
                enc = encodings[idx]
                ids = enc["input_ids"].to(device)
                mask_t = enc["attention_mask"].to(device)
                ttype = enc.get("token_type_ids")
                if ttype is not None:
                    ttype = ttype.to(device)
                logits = m(ids, mask_t, ttype)
                b = logits.float().cpu().numpy()[0]
                bio_sums[idx] = b if bio_sums[idx] is None else bio_sums[idx] + b

        del m
        torch.cuda.empty_cache()

    n_models = len(ckpt_paths)
    rows = []
    for idx in range(n_posts):
        rid = lb_df.iloc[idx]["row_id"]
        text = str(lb_df.iloc[idx]["post"])
        if indicator_mask[idx]:
            rows.append({"row_id": rid, "evidence": "none"})
            continue
        bio_preds = np.argmax(bio_sums[idx] / n_models, axis=-1)
        spans = bio_to_spans(bio_preds, offsets_list[idx], text)
        ev_str = "; ".join(s for s in spans if s.lower() != "none") or "none"
        rows.append({"row_id": rid, "evidence": ev_str})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    n_ev = (out["evidence"] != "none").sum()
    print(f"\nSaved {len(out)} rows → {out_path}")
    print(f"Posts with evidence: {n_ev}/{len(out)} ({n_ev/len(out):.1%})")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                   help="Output directories to ensemble (relative to this file or absolute)")
    p.add_argument("--risk-csv", default=None,
                   help="Path to 1a risk predictions CSV (indicator posts → 'none')")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: evidence_predictions_Nway.csv)")
    args = p.parse_args()
    predict(dirs=args.dirs, risk_csv=args.risk_csv, out_path=args.out)
