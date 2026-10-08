"""Retroactive test: would importance-weighted validation have predicted the
real Sub1b regression (the 4-way composition submission, implied real Phrase
F1 delta ~ -0.0196)?

Reuses the SAME importance weights computed from exp09's embedding space
(population_shift_diagnostic.py / importance_weighted_validation.py) --
these weights are a property of the post text itself ("does this training
post look like a leaderboard post"), not tied to which subtask's model
evaluates it. Both 1a and 1b read the same train.xlsx with no reordering, so
the weights array aligns by row position directly.

We don't have v37's actual original 1b checkpoint locally (referenced
elsewhere as a CRF+seed100 variant never located on this machine), so this
can't reproduce the exact real-vs-predicted comparison the way the 1a check
did. Instead it tests the directional question that's actually available:
does the 4-way composition's advantage over a simple single-family baseline
(`denoised` alone) shrink or flip once true-OOF Phrase F1 is reweighted by
the same population-shift correction -- consistent with (though not proof
of) the real regression observed.

Usage:
    python importance_weighted_validation_1b.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

from config import *
from dataset import load_train, get_tokenizer
from model import EvidenceExtractor
from crf import CRF
from train import bio_to_spans, phrase_f1_single

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "subtask_1a_risk_classification", "results",
                        "exp09_deberta_large_lr_high", "sequence_data")

FAMILIES_TO_TEST = {
    "denoised (solo baseline)": [("denoised", "outputs_denoised", False)],
    "4-way composition (SUBMITTED, real regression)": [
        ("vanilla", "outputs", False),
        ("crf", "outputs_crf", True),
        ("denoised", "outputs_denoised", False),
        ("crf_denoised", "outputs_crf_denoised", True),
    ],
}


def fit_adversarial_weights():
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))
    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit((X - mu) / sigma, y)
    p_lb = clf.predict_proba((train_emb - mu) / sigma)[:, 1]
    p_lb = np.clip(p_lb, 0.02, 0.98)
    w = p_lb / (1 - p_lb)
    w = w / w.mean()
    return np.clip(w, 0.1, 10.0)


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


def average_crf_states(states):
    keys = states[0].keys()
    avg = {k: torch.stack([sd[k] for sd in states]).mean(0) for k in keys}
    crf = CRF(NUM_BIO).to(DEVICE)
    crf.load_state_dict(avg)
    crf.eval()
    return crf


@torch.no_grad()
def get_composition_per_row_pf1(members, tokenizer, orig_df, splits):
    """members: list of (name, dir, has_crf). Returns array of per-row true-OOF phrase F1,
    aligned to orig_df's row order."""
    row_pf1 = np.zeros(len(orig_df))

    for fold, (tr_i, va_i) in enumerate(splits):
        va_df = orig_df.iloc[va_i]
        member_logits = []
        crf_states = []
        for name, fam_dir, has_crf in members:
            model, crf = load_member(fam_dir, fold, has_crf)
            fold_logits = []
            for _, row in va_df.iterrows():
                text = str(row["post"])
                enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True,
                                return_tensors="pt", return_offsets_mapping=True)
                ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
                logits = model(ids, mask).float()
                fold_logits.append(logits.cpu())
            member_logits.append(torch.cat(fold_logits, dim=0))
            if crf is not None:
                crf_states.append({k: v.clone().cpu() for k, v in crf.state_dict().items()})
            del model
            if crf is not None:
                del crf
            torch.cuda.empty_cache()

        avg_logits = torch.stack(member_logits).mean(0)  # (N, seq, NUM_BIO)
        crf_decoder = average_crf_states(crf_states) if crf_states else None

        for i, (_, row) in enumerate(va_df.iterrows()):
            text = str(row["post"])
            enc = tokenizer(text, max_length=MAX_LENGTH, padding="max_length", truncation=True,
                            return_tensors="pt", return_offsets_mapping=True)
            mask = enc["attention_mask"].to(DEVICE)
            logits_i = avg_logits[i:i+1].to(DEVICE)
            if crf_decoder is not None:
                bio = crf_decoder.decode(logits_i, mask.bool())[0]
            else:
                bio = logits_i.argmax(-1).cpu().numpy()[0]
            pred_spans = bio_to_spans(bio, enc["offset_mapping"].squeeze(0).tolist(), text)
            true_spans = row.get("evidence_list", []) or ["none"]
            _, _, f1 = phrase_f1_single(pred_spans, true_spans)
            row_pf1[va_i[i]] = f1

        print(f"    fold {fold} done")

    return row_pf1


def main():
    print(f"{'='*70}\n  Fitting adversarial-validation importance weights (reused from 1a)\n{'='*70}")
    weights = fit_adversarial_weights()
    print(f"  weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

    tokenizer = get_tokenizer()
    df = load_train(oversample_source=None)
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    groups = orig_df["anon_user_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(orig_df, groups=groups))
    assert len(orig_df) == len(weights), f"row count mismatch: {len(orig_df)} vs {len(weights)}"

    results = {}
    for name, members in FAMILIES_TO_TEST.items():
        print(f"\n{'='*70}\n  {name}\n{'='*70}")
        row_pf1 = get_composition_per_row_pf1(members, tokenizer, orig_df, splits)
        unweighted = row_pf1.mean()
        weighted = np.average(row_pf1, weights=weights)
        results[name] = (unweighted, weighted)
        print(f"  unweighted true-OOF Phrase F1: {unweighted:.4f}")
        print(f"  IMPORTANCE-WEIGHTED Phrase F1: {weighted:.4f}")

    baseline_name = "denoised (solo baseline)"
    comp_name = "4-way composition (SUBMITTED, real regression)"
    bu, bw = results[baseline_name]
    cu, cw = results[comp_name]
    print(f"\n{'='*70}\n  DELTA: composition vs solo baseline\n{'='*70}")
    print(f"  unweighted delta: {cu - bu:+.4f}  (this is what the composition search actually reported)")
    print(f"  weighted delta:   {cw - bw:+.4f}  (would this have looked less promising?)")
    if (cu - bu) > 0 and (cw - bw) < (cu - bu):
        print("  -> Reweighting reduced the composition's apparent advantage" +
              (" and flipped it negative" if (cw - bw) < 0 else " (didn't flip, but less misleading)"))
    else:
        print("  -> Reweighting did not reduce the composition's apparent advantage")


if __name__ == "__main__":
    main()
