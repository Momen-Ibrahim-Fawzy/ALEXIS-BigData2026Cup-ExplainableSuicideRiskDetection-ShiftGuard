"""
Sub1a as text-to-text QA, flan-t5-xl (2.85B) via LoRA.

Follow-up to the flan-t5-large line (base 0.7493, large-bare 0.7784,
large-definitions 0.7762, large-768ctx-confounded 0.7504 -- all true-OOF WF1
vs exp09's 0.8014). Capacity scaling from base->large closed a meaningful
chunk of the gap (+0.0291); this tests whether it keeps closing at the next
size up.

Why LoRA, not full fine-tuning: flan-t5-large (783M) already needed ~14GB for
full fine-tuning and OOM'd twice on this shared, contended GPU even at a small
batch size. flan-t5-xl is 2.85B (~3.65x larger) -- full fine-tuning would need
roughly proportionally more (~50GB+), unlikely to fit reliably alongside other
users' jobs. LoRA freezes the 3B backbone and trains only small adapter
matrices (~2% of params), which should fit comfortably. Also a genuinely
different regularization regime than anything tried before in this project
(frozen backbone = can't overfit backbone weights to this ~1600-example
dataset the way full fine-tuning can).

Evaluation: scored-decoding (teacher-forced log-prob over the 4 candidate
labels, argmax) from the start, not greedy generation -- already verified
byte-identical to generation for flan-t5-large (0.0000 delta on every fold),
so no need to re-verify that equivalence here; scoring is simpler to get right
in a training loop (no generation-loop batching complexity).

Model source: google/flan-t5-xl, copied from a verified-working
extraction found in another user's home directory on this shared machine
(same due-diligence as flan-t5-large: loaded, ran a translation + a
domain-relevant zero-shot classification test, got correct/plausible output on
both, before use). No network access involved.

Usage:
    python exp_flant5_xl_qa_sft_lora.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                           get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, RISK2ID, ID2RISK, DATA_DIR, RESULTS_DIR

EXP_NAME = "exp_flant5_xl_qa_sft_lora"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "google/flan-t5-xl"
MAX_SRC_LEN = 512        # matches the flan-t5-large bare-prompt baseline exactly (direct comparability)
MAX_TGT_LEN = 6
N_FOLDS = 5
EPOCHS = 15
PATIENCE = 3
LR = 1e-4                # LoRA typically wants a higher LR than full fine-tuning (fewer, better-conditioned params)
BATCH_SIZE = 2           # reduced further -- batch_size=4 OOM'd by only ~32MB on a
                          # heavily shared GPU (exp43 training + 2 other users' VLLM
                          # engines), so trading more speed for a comfortable margin
GRAD_ACCUM = 8            # increased to compensate -- effective batch still 16
WARMUP_RATIO = 0.06
SEED = 42
CLASS_WEIGHTS = {"indicator": 0.669, "ideation": 0.788, "behavior": 1.045, "attempt": 3.586}

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q", "v", "k", "o", "wi", "wo"]  # standard T5 attention + FFN

PROMPT_TEMPLATE = (
    "Classify the suicide risk level expressed in the following social media "
    "post into exactly one of these four levels, ordered from lowest to "
    "highest risk: indicator, ideation, behavior, attempt.\n\n"
    "Post: {post}\n\n"
    "Risk level:"
)


class QADataset(Dataset):
    def __init__(self, df, tokenizer):
        self.posts = df["post"].astype(str).tolist()
        self.labels = df["risk_norm"].tolist()
        self.weights = [CLASS_WEIGHTS[l] for l in self.labels]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, i):
        prompt = PROMPT_TEMPLATE.format(post=self.posts[i])
        src = self.tokenizer(prompt, max_length=MAX_SRC_LEN, padding="max_length",
                              truncation=True, return_tensors="pt")
        tgt = self.tokenizer(self.labels[i], max_length=MAX_TGT_LEN, padding="max_length",
                              truncation=True, return_tensors="pt")
        labels = tgt["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": labels,
            "sample_weight": torch.tensor(self.weights[i], dtype=torch.float32),
        }


def per_example_loss(logits, labels):
    B, T, V = logits.shape
    loss_flat = nn.functional.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(B, T)
    valid = (labels != -100).float()
    return (loss_flat * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


@torch.no_grad()
def scored_predict(model, tokenizer, posts, device, batch_size=16):
    """Score all 4 candidate labels via teacher-forced log-prob, return argmax preds.
    Same methodology already verified equivalent to generation for flan-t5-large."""
    model.eval()
    prompts = [PROMPT_TEMPLATE.format(post=p) for p in posts]
    label_ids = {label: tokenizer(label, return_tensors="pt")["input_ids"][0] for label in RISK_LEVELS}
    all_scores = np.zeros((len(prompts), len(RISK_LEVELS)))

    for li, label in enumerate(RISK_LEVELS):
        tgt_ids = label_ids[label]
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            enc = tokenizer(batch_prompts, max_length=MAX_SRC_LEN, padding=True,
                             truncation=True, return_tensors="pt").to(device)
            bs = len(batch_prompts)
            decoder_input = tgt_ids.unsqueeze(0).repeat(bs, 1).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=decoder_input)
            log_probs = torch.log_softmax(out.logits.float(), dim=-1)
            tgt = decoder_input.unsqueeze(-1)
            token_lp = torch.gather(log_probs, 2, tgt).squeeze(-1)
            seq_lp = token_lp.sum(dim=1).cpu().numpy()
            all_scores[i:i+bs, li] = seq_lp

    return all_scores.argmax(axis=1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-name", type=str, default=EXP_NAME,
                        help="Distinct results/<name>/ dir -- use for seed-bagging so "
                             "the seed=42 run's checkpoints/summary are never overwritten")
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = os.path.join(RESULTS_DIR, args.output_name)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model: {MODEL_NAME}  (LoRA r={LORA_R}, alpha={LORA_ALPHA})  seed={args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    print(f"Dataset: {len(df)} posts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values
    all_preds = np.zeros(len(df), dtype=int)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n{'='*60}\n  FOLD {fold+1}/{N_FOLDS}\n{'='*60}")
        tr_df = df.iloc[tr_i].reset_index(drop=True)
        va_df = df.iloc[va_i].reset_index(drop=True)

        tr_ds = QADataset(tr_df, tokenizer)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(base_model, lora_config).to(device)
        if fold == 0:
            model.print_trainable_parameters()

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
        total_steps = len(tr_ld) * EPOCHS // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        best_wf1, best_adapter_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for step, batch in enumerate(tr_ld):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                sw = batch["sample_weight"].to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, attention_mask=mask, labels=labels)
                per_ex = per_example_loss(out.logits.float(), labels)
                loss = (per_ex * sw).sum() / sw.sum() / GRAD_ACCUM
                loss.backward()

                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            preds_id = scored_predict(model, tokenizer, va_df["post"].tolist(), device)
            va_true = np.array([RISK2ID[r] for r in va_df["risk_norm"]])
            wf1 = f1_score(va_true, preds_id, average="weighted", zero_division=0)
            print(f"  Epoch {epoch+1}/{EPOCHS}  val WF1={wf1:.4f}")

            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_adapter_state = {k: v.clone().cpu() for k, v in model.state_dict().items() if "lora" in k}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_adapter_state, strict=False)
        print(f"  Fold {fold}: best val WF1={best_wf1:.4f}")

        final_preds = scored_predict(model, tokenizer, va_df["post"].tolist(), device)
        all_preds[va_i] = final_preds

        fold_dir = os.path.join(OUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.save_pretrained(fold_dir)  # LoRA adapter only, small (~tens of MB)

        del model, base_model, optimizer, scheduler
        torch.cuda.empty_cache()

    oof_wf1 = f1_score(true_labels, all_preds, average="weighted", zero_division=0)
    oof_mf1 = f1_score(true_labels, all_preds, average="macro", zero_division=0)
    print(f"\n{'='*60}\n  TRUE OOF RESULTS ({N_FOLDS}-fold)\n{'='*60}")
    print(f"  WF1: {oof_wf1:.4f}")
    print(f"  MF1: {oof_mf1:.4f}")
    print(f"  exp09 baseline WF1: 0.8014 (true OOF)")
    print(f"  vs exp09: {oof_wf1 - 0.8014:+.4f}")
    print(f"  flan-t5-large (bare, 512ctx) WF1: 0.7784 (true OOF)")
    print(f"  vs flan-t5-large: {oof_wf1 - 0.7784:+.4f}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "experiment": EXP_NAME, "model": MODEL_NAME, "method": "LoRA",
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "oof_wf1": float(oof_wf1), "oof_mf1": float(oof_mf1),
            "exp09_baseline_wf1": 0.8014, "improvement_vs_exp09": float(oof_wf1 - 0.8014),
            "flant5_large_bare_wf1": 0.7784, "improvement_vs_flant5_large": float(oof_wf1 - 0.7784),
        }, f, indent=2)


if __name__ == "__main__":
    main()
