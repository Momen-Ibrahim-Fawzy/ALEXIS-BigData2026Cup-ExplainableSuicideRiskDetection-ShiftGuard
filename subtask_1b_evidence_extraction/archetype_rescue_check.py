"""
Theory-driven archetype signal for Sub1b (from BACKLOG.md, CLPsych 2024
"Archetypes and Entropy" -- combining theory-driven prototype statements with
learned model signal beat either alone for evidence extraction).

Cheap first test: POST-PROCESSING ONLY, no retraining. Uses the already-
computed denoised-family true OOF predictions. For posts where the tagger
predicted "none" but the true label has evidence, check whether a sentence
in the post scores highly against Joiner's Interpersonal Theory prototypes
(thwarted belongingness, perceived burdensomeness, acquired capability,
direct/ambivalent desire to die) -- if so, "rescue" that sentence as an
additional candidate span, and re-score true OOF phrase F1.

Directly targets the negation-pattern finding from error_analysis.py
("don't wanna die" doesn't look like direct desire-to-die surface language
to a token tagger, but should score highly against a desire-to-die/
ambivalence archetype prototype semantically).

Usage:
    python archetype_rescue_check.py
"""

import os, sys, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(__file__))
from train import phrase_f1_single, normalize

MODEL_NAME = "microsoft/deberta-v3-large"

PROTOTYPES = {
    "thwarted_belongingness": [
        "I feel completely alone and disconnected from everyone.",
        "Nobody would even notice or care if I was gone.",
        "I don't belong anywhere and no one understands me.",
    ],
    "perceived_burdensomeness": [
        "I am a burden to my family and everyone around me.",
        "Everyone would be better off without me.",
        "My death would be a relief to the people who know me.",
    ],
    "acquired_capability": [
        "I'm not afraid to die anymore.",
        "I've thought carefully about how I would do it.",
        "Pain doesn't scare me the way it used to.",
    ],
    "desire_to_die": [
        "I want to die.",
        "I wish I was dead.",
        "I don't want to be alive anymore.",
        "It wouldn't matter if I was dead.",
        "I don't want to wake up tomorrow.",
    ],
}


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, batch_size=32):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=64, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        all_emb.append(pooled.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def split_sentences(text):
    # simple sentence/clause split -- good enough for short social-media posts
    parts = re.split(r'(?<=[.!?\n])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def main():
    device = torch.device("cuda")
    print("Loading DeBERTa-v3-large for sentence embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).float().eval()

    all_prototypes = [p for plist in PROTOTYPES.values() for p in plist]
    proto_emb = embed_texts(all_prototypes, tokenizer, model, device)
    print(f"Embedded {len(all_prototypes)} prototype statements across {len(PROTOTYPES)} constructs")

    df = pd.read_csv("/tmp/1b_denoised_oof_errors.csv")
    missed = df[(df.true_spans != "none") & (df.pred_spans == "none")].copy()
    print(f"\nMissed-evidence cases to test rescue on: {len(missed)}")

    # Baseline: current F1 on this subset (all 0.0, since pred=none but true!=none)
    baseline_f1 = missed.apply(lambda r: phrase_f1_single(["none"], r.true_spans.split("; ")), axis=1)
    baseline_mean = np.mean([f1 for _, _, f1 in baseline_f1])
    print(f"Baseline mean F1 on this subset (all predict 'none'): {baseline_mean:.4f}")

    # Archetype rescue: for each missed post, split into sentences, embed, compare to prototypes
    rescued_spans = []
    for _, row in missed.iterrows():
        sentences = split_sentences(str(row.post))
        if not sentences:
            rescued_spans.append(["none"])
            continue
        sent_emb = embed_texts(sentences, tokenizer, model, device)
        sims = sent_emb @ proto_emb.T  # (n_sent, n_proto)
        max_sim_per_sent = sims.max(axis=1)
        best_idx = max_sim_per_sent.argmax()
        best_sim = max_sim_per_sent[best_idx]
        if best_sim > 0.75:  # threshold, tuned qualitatively below
            rescued_spans.append([sentences[best_idx]])
        else:
            rescued_spans.append(["none"])

    missed["rescued_span"] = rescued_spans
    rescued_f1 = missed.apply(lambda r: phrase_f1_single(r.rescued_span, r.true_spans.split("; ")), axis=1)
    rescued_mean = np.mean([f1 for _, _, f1 in rescued_f1])
    n_rescued = sum(1 for s in rescued_spans if s != ["none"])
    print(f"\nRescued {n_rescued}/{len(missed)} posts with a candidate span (similarity > 0.75)")
    print(f"Mean F1 on this subset after rescue: {rescued_mean:.4f}")
    print(f"Improvement on this subset: {rescued_mean - baseline_mean:+.4f}")

    # Show a few rescued examples for sanity
    print("\nSample rescued spans:")
    shown = 0
    for i, (_, row) in enumerate(missed.iterrows()):
        if rescued_spans[i] != ["none"] and shown < 8:
            print(f"  true={row.true_spans!r}")
            print(f"  rescued={rescued_spans[i]!r}")
            print()
            shown += 1


if __name__ == "__main__":
    main()
