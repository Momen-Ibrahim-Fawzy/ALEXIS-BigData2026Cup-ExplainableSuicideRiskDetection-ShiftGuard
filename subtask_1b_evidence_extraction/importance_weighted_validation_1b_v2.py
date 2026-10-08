"""Importance-weighted validation for Sub1b, v2 -- uses adversarial weights
fit on Sub1b's OWN embedding space (denoised model, AUC=0.7172) instead of
borrowed Sub1a classification embeddings (AUC=0.656), which failed to flag
either known real Sub1b regression (composition: -0.0196, denoised-solo:
-0.0250 Phrase F1). Now has TWO known real outcomes to check against instead
of one, both negative.

Usage:
    python importance_weighted_validation_1b_v2.py
"""

import os
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

FAMILIES_TO_TEST = {
    "denoised (solo, REAL: -0.0250)": [("denoised", "outputs_denoised", False)],
    "4-way composition (REAL: -0.0196)": [
        ("vanilla", "outputs", False),
        ("crf", "outputs_crf", True),
        ("denoised", "outputs_denoised", False),
        ("crf_denoised", "outputs_crf_denoised", True),
    ],
    "3-way no-denoised (UNTESTED, local 0.8720)": [
        ("vanilla", "outputs", False),
        ("crf", "outputs_crf", True),
        ("crf_denoised", "outputs_crf_denoised", True),
    ],
    "2-way vanilla+crf (UNTESTED, local 0.8669)": [
        ("vanilla", "outputs", False),
        ("crf", "outputs_crf", True),
    ],
    "2-way crf+denoised (UNTESTED, local 0.8671)": [
        ("crf", "outputs_crf", True),
        ("denoised", "outputs_denoised", False),
    ],
}


def fit_adversarial_weights_1b_native():
    train_emb = np.load("train_1b_embeddings_denoised_fold0.npy")
    lb_emb = np.load("leaderboard_1b_embeddings_denoised_fold0.npy")
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
def get_composition_per_row_pf1(name, members, tokenizer, orig_df, splits):
    safe_name = name.split(" (")[0].replace(" ", "_").replace(",", "")
    cache_path = f"row_pf1_{safe_name}.npy"
    if os.path.exists(cache_path):
        print(f"    (using cached {cache_path})")
        return np.load(cache_path)

    row_pf1 = np.zeros(len(orig_df))
    for fold, (tr_i, va_i) in enumerate(splits):
        va_df = orig_df.iloc[va_i]
        member_logits = []
        crf_states = []
        for mname, fam_dir, has_crf in members:
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

        avg_logits = torch.stack(member_logits).mean(0)
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

        print(f"    fold {fold} done", flush=True)

    np.save(cache_path, row_pf1)
    return row_pf1


def main():
    print(f"{'='*70}\n  Fitting adversarial weights on Sub1b's OWN embedding space\n{'='*70}")
    weights = fit_adversarial_weights_1b_native()
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
        row_pf1 = get_composition_per_row_pf1(name, members, tokenizer, orig_df, splits)
        unweighted = row_pf1.mean()
        weighted = np.average(row_pf1, weights=weights)
        results[name] = (unweighted, weighted)
        print(f"  unweighted true-OOF Phrase F1: {unweighted:.4f}")
        print(f"  IMPORTANCE-WEIGHTED Phrase F1: {weighted:.4f}")

    print(f"\n{'='*70}\n  SUMMARY (both candidates, both known to have regressed for real)\n{'='*70}")
    for name, (u, w) in results.items():
        print(f"  {name}: unweighted={u:.4f} weighted={w:.4f} shift={w-u:+.4f}")

    denoised_name = "denoised (solo, REAL: -0.0250)"
    comp_name = "4-way composition (REAL: -0.0196)"
    du, dw = results[denoised_name]
    cu, cw = results[comp_name]
    print(f"\n{'='*70}\n  KEY TEST: does reweighting correctly rank composition ABOVE denoised-solo?\n"
          f"  (real outcome: composition regressed LESS than denoised-solo, -0.0196 vs -0.0250)\n{'='*70}")
    print(f"  unweighted: composition {'beats' if cu>du else 'loses to'} denoised-solo by {cu-du:+.4f}")
    print(f"  weighted:   composition {'beats' if cw>dw else 'loses to'} denoised-solo by {cw-dw:+.4f}")
    if (cw - dw) > (cu - du):
        print("  -> Reweighting moved composition's relative standing UP vs denoised-solo -- "
              "matches the real ordering (composition was the smaller regression)")
    elif (cw - dw) < (cu - du):
        print("  -> Reweighting moved composition's relative standing DOWN vs denoised-solo -- "
              "contradicts the real ordering")
    else:
        print("  -> No change in relative ordering")


if __name__ == "__main__":
    main()
