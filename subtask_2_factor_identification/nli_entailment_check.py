"""Stage 0 of the cross-encoder/entailment idea: zero-shot check, NO
fine-tuning, NO training -- same discipline as validating frozen SBert
before building anything on top of it (mean gap 0.075, all 24 factors
showed signal there).

Uses a DeBERTa-v3-large checkpoint already fine-tuned on ~700k+ NLI pairs
(MNLI+FEVER+ANLI+LingNLI+WANLI) via HuggingFace's zero-shot-classification
pipeline: treats each post as the premise and each factor description as a
hypothesis ("This example is {description}"), scoring entailment
independently per factor (multi_label=True). If this checkpoint's
off-the-shelf entailment reasoning already separates posts that have a
factor from posts that don't -- with ZERO exposure to our data -- that's
strong evidence the cross-encoder/entailment reformulation has real legs
before committing to a fine-tune. If it shows nothing, that's a cheap
warning sign before investing further.

Usage:
    python nli_entailment_check.py [--subset N]
"""

import argparse, ast, time
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

from config import TRAIN_PATH, FACTOR_LIST, FACTOR2ID, NUM_FACTORS
from factor_descriptions import FACTOR_DESCRIPTIONS_BY_NAME

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
HYPOTHESIS_TEMPLATE = "This example shows: {}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                     help="Only run on the first N posts (for a quick sanity check before the full 1635)")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {NLI_MODEL} (device={'cuda' if device==0 else 'cpu'})...")
    classifier = pipeline("zero-shot-classification", model=NLI_MODEL, device=device)

    df = pd.read_excel(TRAIN_PATH, engine="openpyxl")
    df["factors_list"] = df["factors"].apply(lambda x: sorted(set(ast.literal_eval(x))) if pd.notna(x) else [])
    if args.subset:
        df = df.head(args.subset).reset_index(drop=True)
    print(f"Posts: {len(df)}")

    labels = np.zeros((len(df), NUM_FACTORS), dtype=np.float32)
    for i, row in df.iterrows():
        for f in row["factors_list"]:
            fl = f.lower().strip()
            if fl in FACTOR2ID:
                labels[i, FACTOR2ID[fl]] = 1.0

    candidate_descriptions = [FACTOR_DESCRIPTIONS_BY_NAME[f] for f in FACTOR_LIST]

    all_scores = np.zeros((len(df), NUM_FACTORS), dtype=np.float32)
    t0 = time.time()
    for i, text in enumerate(df["post"].astype(str).tolist()):
        result = classifier(text, candidate_descriptions, multi_label=True,
                             hypothesis_template=HYPOTHESIS_TEMPLATE)
        # result['labels']/['scores'] are sorted by score, not by input order -- realign
        score_by_desc = dict(zip(result["labels"], result["scores"]))
        for j, desc in enumerate(candidate_descriptions):
            all_scores[i, j] = score_by_desc[desc]
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(df) - i - 1) / rate
            print(f"  {i+1}/{len(df)}  ({rate:.1f} posts/s, ETA {eta/60:.1f} min)")

    np.save("nli_zeroshot_scores.npy", all_scores)
    np.save("nli_zeroshot_labels.npy", labels)
    print(f"\nSaved -> nli_zeroshot_scores.npy ({all_scores.shape})")

    print(f"\n{'factor':45s} {'score|pos':>10s} {'score|neg':>10s} {'gap':>8s}  n_pos")
    gaps = []
    for i, f in enumerate(FACTOR_LIST):
        pos = all_scores[labels[:, i] == 1, i]
        neg = all_scores[labels[:, i] == 0, i]
        gap = (pos.mean() if len(pos) else float("nan")) - neg.mean()
        gaps.append(gap)
        print(f"{f:45s} {pos.mean() if len(pos) else float('nan'):10.4f} {neg.mean():10.4f} {gap:8.4f}  {int(labels[:,i].sum())}")

    valid_gaps = [g for g in gaps if not np.isnan(g)]
    print(f"\nMean gap across factors: {np.mean(valid_gaps):.4f}")
    print(f"(compare to frozen SBert's mean gap: 0.0754, which validated real before we built on it)")
    print(f"Factors with negative/near-zero gap (<0.01): "
          f"{[FACTOR_LIST[i] for i, g in enumerate(gaps) if not np.isnan(g) and g < 0.01]}")

    if np.mean(valid_gaps) > 0.03:
        print("\n*** Real signal detected -- worth proceeding to a supervised fine-tune "
              "(with GroupKFold-by-user splitting, pseudo-leaderboard validation). ***")
    else:
        print("\n*** Weak/no signal off-the-shelf -- reconsider before investing in a fine-tune. ***")


if __name__ == "__main__":
    main()
