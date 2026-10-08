"""
Sub1a as text-to-text QA, definitions-augmented prompt retry.

Follow-up to exp_flant5_large_qa_sft.py, which scored 0.7784 true-OOF WF1 vs
exp09's 0.8014 (-0.0230) -- capacity-matching (flan-t5-base -> -large) closed
most but not all of the gap. Since this is SFT (not zero-shot), the model
already learns the label mapping from the training pairs regardless of
prompt wording -- so this isn't "phrasing it more nicely", it's adding real
information content the model doesn't otherwise have: a one-line
distinguishing definition for each of the 4 risk levels, instead of just
the bare class names. Definitions are reused verbatim (for consistency with
how this project already characterizes these categories) from
risk_hypotheses.py's ordinal-NLI threshold hypotheses, written earlier for a
different (CORAL/ordinal) reformulation of this same task.

MAX_SRC_LEN raised 512 -> 640: the definitions add ~90 tokens of prompt
overhead (measured via the actual tokenizer), and truncation happens at the
end of the sequence where the post text lives -- without this adjustment,
posts would get truncated more than in the baseline run, an uncontrolled
confound working against whatever the definitions themselves contribute.

Everything else (model, per-example class weighting, training loop,
evaluation) is identical to exp_flant5_large_qa_sft.py -- prompt template and
MAX_SRC_LEN are the only two variables that differ, so this is a clean
single-variable comparison.

Usage:
    python exp_flant5_large_qa_sft_defprompt.py
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import RISK_LEVELS, RISK2ID, ID2RISK, BASE_DIR, DATA_DIR, RESULTS_DIR

EXP_NAME = "exp_flant5_large_qa_sft_defprompt"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "google/flan-t5-large"  # local path, not a hub repo id -- no network needed
MAX_SRC_LEN = 640        # 512 + ~90 tokens of definition overhead, see docstring
MAX_TGT_LEN = 6          # class names are 1-3 subword tokens; generous headroom
N_FOLDS = 5
EPOCHS = 15
PATIENCE = 3
LR = 3e-4                # standard T5 fine-tuning LR (much higher than DeBERTa's 2e-5;
                          # T5's pretraining/architecture calls for a different LR regime)
BATCH_SIZE = 8
GRAD_ACCUM = 2            # effective batch 16, matches exp09
WARMUP_RATIO = 0.06
SEED = 42
CLASS_WEIGHTS = {"indicator": 0.669, "ideation": 0.788, "behavior": 1.045, "attempt": 3.586}

PROMPT_TEMPLATE = (
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
    """(B, T, V) logits, (B, T) labels with -100 padding -> (B,) mean per-token CE per example."""
    B, T, V = logits.shape
    loss_flat = nn.functional.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(B, T)
    valid = (labels != -100).float()
    return (loss_flat * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device):
    model.eval()
    preds = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=MAX_TGT_LEN,
                              num_beams=1, do_sample=False)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend([d.strip().lower() for d in decoded])
    return preds


def parse_pred(text):
    """Map a generated string back to a class id, falling back to nearest valid
    class name by substring match if the model didn't generate an exact match
    (should be rare after fine-tuning, but must not crash if it happens)."""
    text = text.strip().lower()
    if text in RISK2ID:
        return RISK2ID[text]
    for level in RISK_LEVELS:
        if level in text or text in level:
            return RISK2ID[level]
    return RISK2ID["indicator"]  # arbitrary fallback, should essentially never trigger


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model: {MODEL_NAME}")

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    print(f"Dataset: {len(df)} posts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Sanity check: confirm each class name tokenizes distinctly and compactly
    for level in RISK_LEVELS:
        toks = tokenizer(level)["input_ids"]
        print(f"  '{level}' -> {len(toks)} tokens: {toks}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df["anon_user_id"].values
    all_preds = np.zeros(len(df), dtype=int)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n{'='*60}\n  FOLD {fold+1}/{N_FOLDS}\n{'='*60}")
        tr_df, va_df = df.iloc[tr_i].reset_index(drop=True), df.iloc[va_i].reset_index(drop=True)

        tr_ds = QADataset(tr_df, tokenizer)
        va_ds = QADataset(va_df, tokenizer)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.gradient_checkpointing_enable()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(tr_ld) * EPOCHS // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

        best_wf1, best_state, patience_ctr = 0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            steps = 0
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                steps += 1

            preds_text = evaluate(model, va_ld, tokenizer, device)
            preds_id = np.array([parse_pred(p) for p in preds_text])
            va_true = np.array([RISK2ID[r] for r in va_df["risk_norm"]])
            wf1 = f1_score(va_true, preds_id, average="weighted", zero_division=0)
            print(f"  Epoch {epoch+1}/{EPOCHS}  val WF1={wf1:.4f}")

            if wf1 > best_wf1:
                best_wf1, patience_ctr = wf1, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        print(f"  Fold {fold}: best val WF1={best_wf1:.4f}")

        model.eval()
        final_preds_text = evaluate(model, va_ld, tokenizer, device)
        final_preds_id = np.array([parse_pred(p) for p in final_preds_text])
        all_preds[va_i] = final_preds_id

        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"best_fold{fold}.pt"))
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    oof_wf1 = f1_score(true_labels, all_preds, average="weighted", zero_division=0)
    oof_mf1 = f1_score(true_labels, all_preds, average="macro", zero_division=0)
    print(f"\n{'='*60}\n  TRUE OOF RESULTS ({N_FOLDS}-fold)\n{'='*60}")
    print(f"  WF1: {oof_wf1:.4f}")
    print(f"  MF1: {oof_mf1:.4f}")
    print(f"  exp09 baseline WF1: 0.8014 (true OOF)")
    print(f"  vs exp09: {oof_wf1 - 0.8014:+.4f}")
    print(f"  flan-t5-base WF1: 0.7493 (true OOF)")
    print(f"  vs flan-t5-base: {oof_wf1 - 0.7493:+.4f}")
    print(f"  flan-t5-large (bare prompt) WF1: 0.7784 (true OOF)")
    print(f"  vs flan-t5-large (bare prompt): {oof_wf1 - 0.7784:+.4f}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "experiment": EXP_NAME,
            "model": MODEL_NAME,
            "oof_wf1": float(oof_wf1),
            "oof_mf1": float(oof_mf1),
            "exp09_baseline_wf1": 0.8014,
            "improvement_vs_exp09": float(oof_wf1 - 0.8014),
            "flant5_base_wf1": 0.7493,
            "improvement_vs_flant5_base": float(oof_wf1 - 0.7493),
            "flant5_large_bareprompt_wf1": 0.7784,
            "improvement_vs_flant5_large_bareprompt": float(oof_wf1 - 0.7784),
        }, f, indent=2)


if __name__ == "__main__":
    main()
