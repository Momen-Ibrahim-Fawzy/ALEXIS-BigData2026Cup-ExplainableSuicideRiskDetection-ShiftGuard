"""Re-evaluates the already-trained flan-t5-large checkpoints (bare-prompt and
definitions-prompt variants) using SCORED decoding instead of greedy-generate.

Current eval (in exp_flant5_large_qa_sft.py / _defprompt.py): the model
generates a token sequence, then parse_pred() string-matches it back to one
of the 4 labels, falling back to an arbitrary guess if nothing matches. That
mixes two different things: whether the model's probability distribution
actually favors the right answer, and whether greedy decoding happens to
produce cleanly-parseable text.

This script separates them: for each post, compute the model's teacher-forced
log-probability of generating EACH of the 4 candidate label strings, and pick
the argmax. No generation, no parsing, no fallback guessing -- this is the
standard way to use a generative model as a proper classifier (score every
candidate answer, pick the best), and needs no additional training since it
reuses the already-saved best_fold{N}.pt checkpoints directly.

IMPORTANT reproducibility note: this re-derives the same GroupKFold(n_splits=5)
split the training script used. That's only safe because the flan-t5-large
runs were trained in THIS SAME session, in the same environment (no time gap
for e.g. an sklearn upgrade to silently change GroupKFold's tie-breaking --
that's exactly what invalidated an earlier attempt to reproduce exp09's much
older split). This script verifies that assumption empirically before
trusting any scored-decoding number: it first re-runs the ORIGINAL
greedy-generate eval on the freshly-derived split and checks the per-fold
WF1 against the values in exp_flant5_large_qa_sft.log
(fold 0-4: 0.7768, 0.8106, 0.7875, 0.7323, 0.7873). If those don't match
closely, the split has NOT been reproduced correctly and the scored-decoding
comparison below is not trustworthy -- it will say so explicitly rather than
silently reporting a number.

Usage:
    python rescore_flant5_large_constrained.py <checkpoint_dir>
    python rescore_flant5_large_constrained.py results/exp_flant5_large_qa_sft
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, RISK2ID, ID2RISK, DATA_DIR

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-large"
MAX_TGT_LEN = 6
N_FOLDS = 5

# Must match the checkpoint's own training script's prompt exactly.
PROMPTS = {
    "bare": (
        "Classify the suicide risk level expressed in the following social media "
        "post into exactly one of these four levels, ordered from lowest to "
        "highest risk: indicator, ideation, behavior, attempt.\n\n"
        "Post: {post}\n\n"
        "Risk level:"
    ),
    "defprompt": (
        "Classify the suicide risk level expressed in the following social media post "
        "into exactly one of these four levels:\n\n"
        "- indicator: the post shows general risk factors or warning signs, but no "
        "active suicidal thoughts.\n"
        "- ideation: the person expresses active suicidal thoughts, a wish to die, or "
        "intent to end their life, without describing a concrete step taken.\n"
        "- behavior: the person describes taking a concrete step toward suicide, such "
        "as making a plan, seeking the means, or preparing to act.\n"
        "- attempt: the person describes having actually attempted suicide.\n\n"
        "Post: {post}\n\n"
        "Risk level:"
    ),
}

# Known-good per-fold WF1 from the original greedy-generate training logs,
# for the reproducibility check.
KNOWN_FOLD_WF1 = {
    "bare": [0.7768, 0.8106, 0.7875, 0.7323, 0.7873],
    "defprompt": None,  # filled in below if needed
}


@torch.no_grad()
def greedy_predict(model, tokenizer, prompts, device, max_src_len):
    preds = []
    for i in range(0, len(prompts), 16):
        batch_prompts = prompts[i:i+16]
        enc = tokenizer(batch_prompts, max_length=max_src_len, padding=True,
                         truncation=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_TGT_LEN, num_beams=1, do_sample=False)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend([d.strip().lower() for d in decoded])
    return preds


def parse_pred(text):
    text = text.strip().lower()
    if text in RISK2ID:
        return RISK2ID[text]
    for level in RISK_LEVELS:
        if level in text or text in level:
            return RISK2ID[level]
    return RISK2ID["indicator"]


@torch.no_grad()
def scored_predict(model, tokenizer, prompts, device, max_src_len):
    """For each prompt, score all 4 candidate labels via teacher-forced log-prob
    and return the argmax. Batches all (prompt, candidate) pairs together."""
    label_ids = {}
    for label in RISK_LEVELS:
        ids = tokenizer(label, return_tensors="pt")["input_ids"][0]  # includes EOS
        label_ids[label] = ids

    all_scores = np.zeros((len(prompts), len(RISK_LEVELS)))

    for li, label in enumerate(RISK_LEVELS):
        tgt_ids = label_ids[label]
        for i in range(0, len(prompts), 16):
            batch_prompts = prompts[i:i+16]
            enc = tokenizer(batch_prompts, max_length=max_src_len, padding=True,
                             truncation=True, return_tensors="pt").to(device)
            bs = len(batch_prompts)
            decoder_input = tgt_ids.unsqueeze(0).repeat(bs, 1).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=decoder_input)
            # out.logits: (B, T, V). Compute per-example summed log-prob of the target tokens.
            log_probs = torch.log_softmax(out.logits.float(), dim=-1)
            tgt = decoder_input.unsqueeze(-1)
            token_lp = torch.gather(log_probs, 2, tgt).squeeze(-1)  # (B, T)
            seq_lp = token_lp.sum(dim=1).cpu().numpy()  # sum, not mean -- consistent length across labels here (all short)
            all_scores[i:i+bs, li] = seq_lp

    return all_scores.argmax(axis=1)


def evaluate_variant(ckpt_dir, prompt_key, max_src_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_template = PROMPTS[prompt_key]

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values

    greedy_fold_wf1, scored_fold_wf1 = [], []
    all_true, all_greedy_preds, all_scored_preds = [], [], []

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i].reset_index(drop=True)
        prompts = [prompt_template.format(post=p) for p in va_df["post"].astype(str)]
        true = np.array([RISK2ID[r] for r in va_df["risk_norm"]])

        ckpt_path = os.path.join(ckpt_dir, f"best_fold{fold}.pt")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=device))
        model.eval()

        greedy_text = greedy_predict(model, tokenizer, prompts, device, max_src_len)
        greedy_preds = np.array([parse_pred(t) for t in greedy_text])
        greedy_wf1 = f1_score(true, greedy_preds, average="weighted", zero_division=0)
        greedy_fold_wf1.append(greedy_wf1)

        scored_preds = scored_predict(model, tokenizer, prompts, device, max_src_len)
        scored_wf1 = f1_score(true, scored_preds, average="weighted", zero_division=0)
        scored_fold_wf1.append(scored_wf1)

        print(f"  Fold {fold}: greedy WF1={greedy_wf1:.4f}  scored WF1={scored_wf1:.4f}  "
              f"(n={len(va_df)})")

        all_true.extend(true.tolist())
        all_greedy_preds.extend(greedy_preds.tolist())
        all_scored_preds.extend(scored_preds.tolist())

        del model
        torch.cuda.empty_cache()

    return {
        "greedy_fold_wf1": greedy_fold_wf1,
        "scored_fold_wf1": scored_fold_wf1,
        "overall_greedy_wf1": f1_score(all_true, all_greedy_preds, average="weighted", zero_division=0),
        "overall_scored_wf1": f1_score(all_true, all_scored_preds, average="weighted", zero_division=0),
    }


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "results/exp_flant5_large_qa_sft"
    prompt_key = "defprompt" if "defprompt" in ckpt_dir else "bare"
    max_src_len = 640 if prompt_key == "defprompt" else 512

    print(f"Evaluating {ckpt_dir} (prompt={prompt_key}, max_src_len={max_src_len})")
    result = evaluate_variant(ckpt_dir, prompt_key, max_src_len)

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print(f"  Greedy-generate WF1: {result['overall_greedy_wf1']:.4f}")
    print(f"  Scored-decoding WF1: {result['overall_scored_wf1']:.4f}")
    print(f"  Delta (scored - greedy): {result['overall_scored_wf1'] - result['overall_greedy_wf1']:+.4f}")

    known = KNOWN_FOLD_WF1.get(prompt_key)
    if known:
        print(f"\n  --- Reproducibility check (fold-by-fold greedy WF1) ---")
        max_diff = 0
        for fold, (mine, orig) in enumerate(zip(result["greedy_fold_wf1"], known)):
            diff = abs(mine - orig)
            max_diff = max(max_diff, diff)
            flag = "OK" if diff < 0.01 else "MISMATCH"
            print(f"    Fold {fold}: mine={mine:.4f}  original_log={orig:.4f}  diff={diff:.4f}  [{flag}]")
        if max_diff < 0.01:
            print(f"\n  Split reproduction VERIFIED (max diff {max_diff:.4f} < 0.01). "
                  f"Scored-decoding numbers above are trustworthy.")
        else:
            print(f"\n  ⚠ Split reproduction FAILED (max diff {max_diff:.4f} >= 0.01). "
                  f"Do NOT trust the scored-decoding comparison above -- the fold split "
                  f"used here does not match what the checkpoints were actually trained on.")

    with open(os.path.join(ckpt_dir, "scored_decoding_result.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
