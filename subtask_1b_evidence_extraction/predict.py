"""Subtask 1b: Generate leaderboard evidence predictions."""

import os, json, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from config import *
from dataset import load_leaderboard, get_tokenizer, EvidenceDataset
from model import EvidenceExtractor
from crf import CRF
from train import bio_to_spans


def predict(risk_csv=None, output_dir=None, use_crf=False):
    """Predict evidence. If risk_csv provided, use it to skip indicator posts."""
    if output_dir:
        globals()["OUTPUT_DIR"] = os.path.abspath(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    lb_df = load_leaderboard()

    risk_map = {}
    if risk_csv and os.path.exists(risk_csv):
        rdf = pd.read_csv(risk_csv)
        risk_map = dict(zip(rdf["row_id"], rdf["risk_level"].str.lower()))
        print(f"Using risk predictions from {risk_csv}")

    model_paths = [os.path.join(OUTPUT_DIR, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS)
                   if os.path.exists(os.path.join(OUTPUT_DIR, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} models")

    models, crfs = [], []
    for p in model_paths:
        ckpt = torch.load(p, weights_only=False, map_location=device)
        m = EvidenceExtractor().to(device).float()
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
        if use_crf:
            if ckpt.get("crf") is None:
                raise ValueError(f"{p} has no saved CRF state -- was this fold trained with --use-crf?")
            c = CRF(NUM_BIO).to(device)
            c.load_state_dict(ckpt["crf"])
            crfs.append(c)

    # Ensemble the CRF by averaging fold transition/start/end params -- mirrors
    # how emissions are already averaged across folds below, so decoding uses
    # one consistent (averaged) transition structure rather than picking one
    # fold's transitions arbitrarily.
    ensembled_crf = None
    if use_crf:
        ensembled_crf = CRF(NUM_BIO).to(device)
        with torch.no_grad():
            ensembled_crf.start_transitions.copy_(torch.stack([c.start_transitions for c in crfs]).mean(0))
            ensembled_crf.end_transitions.copy_(torch.stack([c.end_transitions for c in crfs]).mean(0))
            ensembled_crf.transitions.copy_(torch.stack([c.transitions for c in crfs]).mean(0))

    rows = []
    for idx in range(len(lb_df)):
        row = lb_df.iloc[idx]
        rid = row["row_id"]
        text = str(row["post"])

        # Skip indicator posts
        if risk_map.get(rid, "") == "indicator":
            rows.append({"row_id": rid, "evidence": "none"})
            continue

        enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                        truncation=True, return_tensors="pt", return_offsets_mapping=True)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        ttype = enc.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        bio_sum = None
        with torch.no_grad():
            for m in models:
                logits = m(ids, mask, ttype)
                b = logits.float().cpu().numpy()[0]
                bio_sum = b if bio_sum is None else bio_sum + b

        avg_emissions = bio_sum / len(models)
        if use_crf:
            avg_t = torch.tensor(avg_emissions, dtype=torch.float32, device=device).unsqueeze(0)
            bio_preds = ensembled_crf.decode(avg_t, mask.bool())[0]
        else:
            bio_preds = np.argmax(avg_emissions, axis=-1)

        spans = bio_to_spans(bio_preds, offsets, text)
        ev_str = "; ".join(s for s in spans if s.lower() != "none") or "none"
        rows.append({"row_id": rid, "evidence": ev_str})

        if (idx+1) % 50 == 0:
            print(f"  {idx+1}/{len(lb_df)}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "evidence_predictions.csv"), index=False)
    has_ev = (out["evidence"] != "none").sum()
    print(f"Saved. Posts with evidence: {has_ev}/{len(out)}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--risk-csv", default=None, help="Path to risk predictions CSV")
    p.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR")
    p.add_argument("--use-crf", action="store_true",
                    help="Must match whether the loaded checkpoints were trained with --use-crf")
    args = p.parse_args()
    predict(risk_csv=args.risk_csv, output_dir=args.output_dir, use_crf=args.use_crf)
