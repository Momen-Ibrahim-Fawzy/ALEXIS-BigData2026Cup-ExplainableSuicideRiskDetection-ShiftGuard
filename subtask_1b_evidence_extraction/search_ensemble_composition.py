"""Sub1b: systematic ensemble-composition search across all available model
families, via true OOF -- not just "add more diversity."

Motivated directly by the remote session's own hard-learned lesson: their
v22 (vanilla+CRF+denoised, a different 3-way mix) REGRESSED vs v21's 2-way,
while v37/38 (denoised+CRF_seed100+CRF_denoised, a DIFFERENT 3-way mix) beat
their prior best. Same ingredients, different combination, opposite outcome
-- "model quality trumps ensemble size" only helps if you actually search
for the winning combination rather than assume one.

For each candidate subset of families, per fold: load each member's model
(and CRF layer, if any), run on that fold's held-out validation posts
(never seen by ANY of them during training), average emission logits,
decode via the averaged CRF transition matrix if any CRF member is present
(else plain argmax), score true OOF phrase-F1. Same discipline as
validate_crf_vanilla_ensemble.py, generalized to N families and all
subset sizes.

Usage:
    python search_ensemble_composition.py
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from config import *
from dataset import load_train, get_tokenizer
from model import EvidenceExtractor
from crf import CRF
from train import bio_to_spans, phrase_f1_single

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# family_name -> (output_dir, has_crf)
FAMILIES = {
    "vanilla":      ("outputs", False),
    "crf":          ("outputs_crf", True),
    "denoised":     ("outputs_denoised", False),
    "crf_denoised": ("outputs_crf_denoised", True),
    "denoised_seed100": ("outputs_denoised_seed100_v2", False),
}

PSEUDO_LB_SPLIT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pseudo_leaderboard_users.json")


def load_member(fam_dir, fold, has_crf, device):
    ckpt = torch.load(f"{fam_dir}/fold_{fold}/best.pt", weights_only=False, map_location=device)
    m = EvidenceExtractor().to(device).float()
    m.load_state_dict(ckpt["model"])
    m.eval()
    crf = None
    if has_crf:
        crf = CRF(NUM_BIO).to(device)
        crf.load_state_dict(ckpt["crf"])
        crf.eval()
    return m, crf


@torch.no_grad()
def get_fold_emissions(fam_dir, has_crf, fold, va_df, tokenizer):
    """Returns list of (offsets, text, emission_logits[seq_len, NUM_BIO], crf_or_None) per row."""
    model, crf = load_member(fam_dir, fold, has_crf, DEVICE)
    rows = []
    for _, row in va_df.iterrows():
        text = str(row["post"])
        enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                         truncation=True, return_tensors="pt", return_offsets_mapping=True)
        ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        ttype = enc.get("token_type_ids")
        if ttype is not None:
            ttype = ttype.to(DEVICE)
        offsets = enc["offset_mapping"].squeeze(0).tolist()
        logits = model(ids, mask, ttype).float()  # (1, seq_len, NUM_BIO)
        rows.append({
            "row_id": row["row_id"], "anon_user_id": row["anon_user_id"],
            "offsets": offsets, "text": text, "mask": mask.cpu(),
            "logits": logits.cpu(), "true_spans": row.get("evidence_list", []) or ["none"],
        })
    crf_state = {k: v.clone().cpu() for k, v in crf.state_dict().items()} if crf is not None else None
    del model
    if crf is not None:
        del crf
    torch.cuda.empty_cache()
    return rows, crf_state


def average_crf_states(crf_state_dicts):
    """Average multiple CRF layers' parameters into one (mirrors predict_crf_vanilla_ensemble.py)."""
    keys = crf_state_dicts[0].keys()
    avg = {}
    for k in keys:
        avg[k] = torch.stack([sd[k] for sd in crf_state_dicts]).mean(0)
    crf = CRF(NUM_BIO).to(DEVICE)
    crf.load_state_dict(avg)
    crf.eval()
    return crf


def main():
    tokenizer = get_tokenizer()
    df = load_train(oversample_source=None)
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    groups = orig_df["anon_user_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(orig_df, groups=groups))

    # ── Step 1: for every family, extract true-OOF emissions per fold (once) ──
    fam_rows = {fam: [] for fam in FAMILIES}   # fam -> flat list of per-row dicts across all folds
    fam_crf_states = {fam: [] for fam in FAMILIES}  # fam -> list of 5 CRF state_dicts (if has_crf)

    for fam, (fam_dir, has_crf) in FAMILIES.items():
        print(f"\n=== Extracting true-OOF emissions: {fam} ({fam_dir}) ===")
        for fold, (tr_i, va_i) in enumerate(splits):
            va_df = orig_df.iloc[va_i]
            rows, crf_state = get_fold_emissions(fam_dir, has_crf, fold, va_df, tokenizer)
            fam_rows[fam].extend(rows)
            if crf_state is not None:
                fam_crf_states[fam].append(crf_state)
            print(f"  fold {fold}: {len(rows)} rows")

    # Sanity: all families must cover the same rows in the same order (same GroupKFold split)
    ref_ids = [r["row_id"] for r in fam_rows["vanilla"]]
    for fam in FAMILIES:
        assert [r["row_id"] for r in fam_rows[fam]] == ref_ids, f"{fam} row order mismatch"
    print(f"\nAll {len(FAMILIES)} families cover {len(ref_ids)} rows in identical order.")

    # ── Step 2: search all 2/3/4-way subsets ─────────────────────────────────
    fam_names = list(FAMILIES.keys())
    results = []
    for size in (2, 3, 4):
        for subset in itertools.combinations(fam_names, size):
            has_any_crf = any(FAMILIES[f][1] for f in subset)
            crf_decoder = None
            if has_any_crf:
                all_crf_states = []
                for f in subset:
                    if FAMILIES[f][1]:
                        all_crf_states.extend(fam_crf_states[f])
                crf_decoder = average_crf_states(all_crf_states)

            pf1s = []
            n = len(ref_ids)
            for i in range(n):
                avg_logits = torch.stack([fam_rows[f][i]["logits"] for f in subset]).mean(0).to(DEVICE)
                mask = fam_rows[subset[0]][i]["mask"].to(DEVICE)
                offsets = fam_rows[subset[0]][i]["offsets"]
                text = fam_rows[subset[0]][i]["text"]
                true_spans = fam_rows[subset[0]][i]["true_spans"]

                if crf_decoder is not None:
                    bio = crf_decoder.decode(avg_logits, mask.bool())[0]
                else:
                    bio = avg_logits.argmax(-1).cpu().numpy()[0]

                pred_spans = bio_to_spans(bio, offsets, text)
                _, _, f1 = phrase_f1_single(pred_spans, true_spans)
                pf1s.append(f1)

            mean_pf1 = float(np.mean(pf1s))
            results.append({"subset": subset, "size": size, "naive_pf1": mean_pf1, "per_row_pf1": pf1s})
            print(f"  {'+'.join(subset):45s} (n={size})  true-OOF PhraseF1 = {mean_pf1:.4f}")

    results.sort(key=lambda r: -r["naive_pf1"])
    print(f"\n{'='*70}\nTOP 10 COMPOSITIONS (naive true-OOF)\n{'='*70}")
    for r in results[:10]:
        print(f"  {r['naive_pf1']:.4f}  {'+'.join(r['subset'])}")

    # ── Step 3: honest pseudo-leaderboard check on the top few ───────────────
    meta = pd.DataFrame({"row_id": ref_ids,
                         "anon_user_id": [r["anon_user_id"] for r in fam_rows["vanilla"]]})
    holdout_users = None
    if os.path.exists(PSEUDO_LB_SPLIT_PATH):
        with open(PSEUDO_LB_SPLIT_PATH) as f:
            holdout_users = set(json.load(f)["pseudo_leaderboard_users"])
        is_holdout = meta["anon_user_id"].isin(holdout_users).values
        print(f"\n{'='*70}\nHONEST CHECK (fixed 20 pseudo-leaderboard users) -- top 5 by naive score\n{'='*70}")
        for r in results[:5]:
            pf1_arr = np.array(r["per_row_pf1"])
            h = pf1_arr[is_holdout].mean()
            o = pf1_arr[~is_holdout].mean()
            print(f"  {'+'.join(r['subset']):45s} naive={r['naive_pf1']:.4f}  "
                  f"pseudo-LB(20)={h:.4f}  other(133)={o:.4f}")

    with open("ensemble_composition_search_result.json", "w") as f:
        json.dump([{"subset": r["subset"], "size": r["size"], "naive_pf1": r["naive_pf1"]}
                   for r in results], f, indent=2)
    print("\nSaved -> ensemble_composition_search_result.json")


if __name__ == "__main__":
    main()
