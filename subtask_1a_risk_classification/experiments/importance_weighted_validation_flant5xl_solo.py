"""Retrospective importance-weighted check for a SOLO flan-t5-xl (LoRA)
submission -- never tried for real (only tried blended at w=0.10 with exp09,
which regressed -0.0031 for real). Reloads the 5 fold LoRA adapters, computes
true-OOF scored-decoding probabilities (same methodology as
exp_flant5_xl_qa_sft_lora.py), then compares unweighted vs importance-weighted
true-OOF WF1 using the same population-shift weights used for every other
retrospective check this session.

Usage:
    python importance_weighted_validation_flant5xl_solo.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, RISK2ID, DATA_DIR, RESULTS_DIR

MODEL_NAME = "google/flan-t5-xl"
EXP_NAME = "exp_flant5_xl_qa_sft_lora"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
MAX_SRC_LEN = 512
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_TEMPLATE = (
    "Classify the suicide risk level expressed in the following social media "
    "post into exactly one of these four levels, ordered from lowest to "
    "highest risk: indicator, ideation, behavior, attempt.\n\n"
    "Post: {post}\n\n"
    "Risk level:"
)


def fit_adversarial_weights():
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))
    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit((X - mu) / sigma, y)
    p_lb = np.clip(clf.predict_proba((train_emb - mu) / sigma)[:, 1], 0.02, 0.98)
    w = p_lb / (1 - p_lb)
    w = w / w.mean()
    return np.clip(w, 0.1, 10.0)


@torch.no_grad()
def scored_probs(model, tokenizer, posts, batch_size=16):
    model.eval()
    prompts = [PROMPT_TEMPLATE.format(post=p) for p in posts]
    label_ids = {label: tokenizer(label, return_tensors="pt")["input_ids"][0] for label in RISK_LEVELS}
    all_scores = np.zeros((len(prompts), len(RISK_LEVELS)))

    for li, label in enumerate(RISK_LEVELS):
        tgt_ids = label_ids[label]
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(batch_prompts, max_length=MAX_SRC_LEN, padding=True,
                             truncation=True, return_tensors="pt").to(DEVICE)
            bs = len(batch_prompts)
            decoder_input = tgt_ids.unsqueeze(0).repeat(bs, 1).to(DEVICE)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=decoder_input)
            log_probs = torch.log_softmax(out.logits.float(), dim=-1)
            tgt = decoder_input.unsqueeze(-1)
            token_lp = torch.gather(log_probs, 2, tgt).squeeze(-1)
            seq_lp = token_lp.sum(dim=1).cpu().numpy()
            all_scores[i:i + bs, li] = seq_lp

    # scores are summed log-probs (not normalized log-probs) -- softmax over the
    # 4 candidates converts to a proper probability distribution for weighting/F1 use
    exp_scores = np.exp(all_scores - all_scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def get_flant5xl_oof_probs(df, groups, gkf, tokenizer):
    cache_path = os.path.join(RESULTS_DIR, "flant5xl_solo_train_oof_probs.npy")
    if os.path.exists(cache_path):
        print("  (using cached flan-t5-xl solo true-OOF probs)")
        return np.load(cache_path)

    all_probs = np.zeros((len(df), 4))
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        model = PeftModel.from_pretrained(base_model, fold_dir).to(DEVICE)
        probs = scored_probs(model, tokenizer, va_df["post"].astype(str).tolist())
        all_probs[va_i] = probs
        del model, base_model
        torch.cuda.empty_cache()
        print(f"    flan-t5-xl solo fold {fold} done")

    np.save(cache_path, all_probs)
    return all_probs


def main():
    print(f"{'='*70}\n  Fitting adversarial-validation importance weights\n{'='*70}")
    weights = fit_adversarial_weights()
    print(f"  weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    groups = df["anon_user_id"].values
    gkf = GroupKFold(n_splits=5)

    print(f"\n{'='*70}\n  Computing flan-t5-xl solo true-OOF probs\n{'='*70}")
    probs = get_flant5xl_oof_probs(df, groups, gkf, tokenizer)
    preds = probs.argmax(axis=1)

    u = f1_score(true_labels, preds, average="weighted", zero_division=0)
    w = f1_score(true_labels, preds, average="weighted", sample_weight=weights, zero_division=0)
    print(f"\n{'='*70}\n  RESULT: flan-t5-xl solo (never submitted for real, standalone)\n{'='*70}")
    print(f"  unweighted true-OOF WF1: {u:.4f}  (summary.json says {0.8038:.4f} -- should match)")
    print(f"  IMPORTANCE-WEIGHTED WF1: {w:.4f}")
    print(f"  delta (weighted - unweighted): {w - u:+.4f}")

    exp09_logits = np.load(os.path.join(SEQ_DIR, "train_oof_logits.npy"))
    exp09_probs = np.exp(exp09_logits) / np.exp(exp09_logits).sum(axis=1, keepdims=True)
    exp09_preds = exp09_probs.argmax(axis=1)
    eu = f1_score(true_labels, exp09_preds, average="weighted", zero_division=0)
    ew = f1_score(true_labels, exp09_preds, average="weighted", sample_weight=weights, zero_division=0)
    print(f"\n  exp09 (trusted, seq-rescored) unweighted: {eu:.4f}  weighted: {ew:.4f}")
    du, dw = u - eu, w - ew
    print(f"\n  vs exp09 -- unweighted delta: {du:+.4f}   weighted delta: {dw:+.4f}")
    if du > 0 and dw < 0:
        print("  -> WEIGHTING FLIPS THE SIGN: reweighting says this would likely regress for real.")
    elif du > 0 and dw > 0 and dw < du:
        print("  -> weighting shrinks the apparent gain (didn't flip, but less promising than it looks).")
    elif du > 0 and dw >= du:
        print("  -> weighting does NOT reduce the apparent gain -- no red flag from this check.")
    else:
        print("  -> unweighted was already flat/negative vs exp09.")


if __name__ == "__main__":
    main()
