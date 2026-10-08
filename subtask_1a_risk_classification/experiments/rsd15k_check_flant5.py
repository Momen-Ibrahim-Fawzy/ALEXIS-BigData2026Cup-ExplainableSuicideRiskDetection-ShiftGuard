"""RSD-15K retrospective check, part 2: flan-t5-large (needed for the
flan-t5 10%-blend ensemble candidate) and flan-t5-xl (the solo candidate).

Usage:
    python rsd15k_check_flant5.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, RISK2ID, RESULTS_DIR

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
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


@torch.no_grad()
def scored_probs_one_fold(model, tokenizer, prompts, batch_size=16):
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
    exp_scores = np.exp(all_scores - all_scores.max(axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def run_flant5_large(rsd, prompts, tokenizer_name="google/flan-t5-large"):
    cache_path = os.path.join(RESULTS_DIR, "flant5_large_rsd15k_probs.npy")
    if os.path.exists(cache_path):
        print("flant5_large: cached, skipping")
        return
    print(f"\n{'='*70}\n  flan-t5-large (10%-blend member)\n{'='*70}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    exp_dir = os.path.join(RESULTS_DIR, "exp_flant5_large_qa_sft")
    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_name).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=DEVICE))
        model.eval()
        probs = scored_probs_one_fold(model, tokenizer, prompts)
        fold_probs.append(probs)
        print(f"    fold {fold} done", flush=True)
        del model
        torch.cuda.empty_cache()
    avg = np.mean(fold_probs, axis=0)
    np.save(cache_path, avg)


def run_flant5_xl(rsd, prompts, model_name="google/flan-t5-xl"):
    cache_path = os.path.join(RESULTS_DIR, "flant5xl_solo_rsd15k_probs.npy")
    if os.path.exists(cache_path):
        print("flant5xl_solo: cached, skipping")
        return
    print(f"\n{'='*70}\n  flan-t5-xl solo\n{'='*70}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    exp_dir = os.path.join(RESULTS_DIR, "exp_flant5_xl_qa_sft_lora")
    fold_probs = []
    for fold in range(N_FOLDS):
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        model = PeftModel.from_pretrained(base_model, fold_dir).to(DEVICE)
        model.eval()
        probs = scored_probs_one_fold(model, tokenizer, prompts)
        fold_probs.append(probs)
        print(f"    fold {fold} done", flush=True)
        del model, base_model
        torch.cuda.empty_cache()
    avg = np.mean(fold_probs, axis=0)
    np.save(cache_path, avg)


def main():
    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    prompts = [PROMPT_TEMPLATE.format(post=p) for p in rsd["text"].astype(str)]
    true_labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])

    run_flant5_large(rsd, prompts)
    run_flant5_xl(rsd, prompts)

    for name, path in [("flant5_large", "flant5_large_rsd15k_probs.npy"),
                        ("flant5xl_solo", "flant5xl_solo_rsd15k_probs.npy")]:
        p = os.path.join(RESULTS_DIR, path)
        if os.path.exists(p):
            probs = np.load(p)
            wf1 = f1_score(true_labels, probs.argmax(1), average="weighted", zero_division=0)
            print(f"{name} RSD-15K WF1: {wf1:.4f}")


if __name__ == "__main__":
    main()
