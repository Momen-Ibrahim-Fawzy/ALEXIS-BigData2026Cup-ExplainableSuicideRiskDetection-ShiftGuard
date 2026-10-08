"""Domain-adaptive continued pretraining: MLM-continue DeBERTa-v3-large on
train+leaderboard post text combined (unlabeled, no labels needed or used),
before fine-tuning on Sub1a. Directly targets the measured population shift
(adversarial-validation AUC~0.66 between train and leaderboard text) instead
of another loss/architecture/blend variant on top of the same 153-user
training distribution -- the first genuinely new category tried this session.

Small corpus (1635+378=2013 short posts) -- moderate epoch count (not 20+)
to avoid overfitting/memorizing rather than genuinely adapting; this is
"don't stop pretraining"-style continued MLM, not pretraining from scratch.

Usage:
    python continued_pretrain_mlm.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
                           Trainer, TrainingArguments)

sys.path.insert(0, os.path.dirname(__file__))
from trainer import DATA_DIR, RESULTS_DIR

MODEL_NAME = "microsoft/deberta-v3-large"
OUT_DIR = os.path.join(RESULTS_DIR, "domain_adapted_deberta_mlm")
MAX_LENGTH = 512
EPOCHS = 4
BATCH_SIZE = 8
LR = 5e-6   # much lower than exp09's 2e-5 -- that LR is tuned for fine-tuning
            # an already-good head, not training a freshly-initialized MLM head
            # from scratch (this checkpoint's saved head keys don't match what
            # AutoModelForMaskedLM expects, so the head starts random; the first
            # attempt at 2e-5 diverged, loss 13.7 -> 724 within half an epoch)
MLM_PROB = 0.15
SEED = 42


class TextOnlyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(str(self.texts[i]), max_length=self.max_length,
                              truncation=True, padding="max_length")
        return {k: torch.tensor(v) for k, v in enc.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    train_df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    texts = pd.concat([train_df["post"].astype(str), lb_df["post"].astype(str)]).tolist()
    print(f"Combined corpus: {len(texts)} posts ({len(train_df)} train + {len(lb_df)} leaderboard, unlabeled)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    dataset = TextOnlyDataset(texts, tokenizer, MAX_LENGTH)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints_tmp"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=20,
        save_strategy="no",
        bf16=False,  # fp32 -- avoid bf16 precision contributing to the earlier instability
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    final_dir = os.path.join(OUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved domain-adapted backbone -> {final_dir}")


if __name__ == "__main__":
    main()
