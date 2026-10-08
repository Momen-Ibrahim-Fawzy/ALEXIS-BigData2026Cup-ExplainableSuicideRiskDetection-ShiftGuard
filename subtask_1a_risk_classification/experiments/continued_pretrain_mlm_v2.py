"""Domain-adaptive continued pretraining, v2 -- fixes the collapse found in
the first attempt (continued_pretrain_mlm.py's saved backbone produced an
IDENTICAL output vector for every input, cosine sim 1.0000 across completely
different sentences -- confirmed via direct probe, and explains why the
downstream Sub1a fine-tune (exp45) never got past near-random WF1 in any
fold). Root cause: no validation signal during the MLM phase itself, so a
smoothly-dropping train loss masked the encoder finding a degenerate
shortcut on this small (2013-post), narrow-domain corpus.

Three safeguards added this time:
  1. Freeze embeddings + bottom 20 of 24 DeBERTa-v3-large layers. Only the
     top 4 layers + LM head adapt -- limits how much the encoder's actual
     representations can move, closer to standard "adapt the last few
     layers" domain-adaptation practice than full-model continued pretraining.
  2. A genuine held-out MLM validation split (90/10) with early stopping on
     val MLM loss -- catches collapse (val loss stops tracking train loss,
     or a sanity probe fails) before it happens, rather than trusting a
     fixed epoch count.
  3. A representation-collapse sanity check after training: encode a
     handful of deliberately different sentences and verify their pooled
     embeddings are NOT near-identical before ever saving/using the
     backbone downstream.

Much lower LR (2e-6) than the first attempt's fixed 5e-6, given even that
was apparently enough to erode useful structure over 4 full-model epochs.

Usage:
    python continued_pretrain_mlm_v2.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
                           Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback)

sys.path.insert(0, os.path.dirname(__file__))
from trainer import DATA_DIR, RESULTS_DIR

MODEL_NAME = "microsoft/deberta-v3-large"
OUT_DIR = os.path.join(RESULTS_DIR, "domain_adapted_deberta_mlm_v2")
MAX_LENGTH = 512
MAX_EPOCHS = 15
BATCH_SIZE = 8
LR = 2e-6
MLM_PROB = 0.15
SEED = 42
N_UNFROZEN_LAYERS = 4   # only top 4 of 24 adapt
VAL_FRAC = 0.10

PROBE_SENTENCES = [
    "I feel hopeless and want to die.",
    "I need someone to talk to right now.",
    "The weather is nice today, I went for a walk.",
    "Planning to overdose again tonight.",
]


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


def freeze_bottom_layers(model, n_unfrozen):
    for p in model.deberta.embeddings.parameters():
        p.requires_grad = False
    layers = model.deberta.encoder.layer
    n_total = len(layers)
    for i, layer in enumerate(layers):
        train_this = i >= (n_total - n_unfrozen)
        for p in layer.parameters():
            p.requires_grad = train_this
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"  Frozen embeddings + bottom {n_total - n_unfrozen}/{n_total} layers. "
          f"Trainable: {n_trainable/1e6:.1f}M / {n_total_params/1e6:.1f}M "
          f"({100*n_trainable/n_total_params:.1f}%)")


@torch.no_grad()
def collapse_check(model, tokenizer, device, tag=""):
    """Encode deliberately different sentences, check pooled embeddings aren't
    near-identical. Returns (min pairwise cosine distance, max pairwise cosine sim)."""
    model.eval()
    base = model.deberta if hasattr(model, "deberta") else model
    enc = tokenizer(PROBE_SENTENCES, return_tensors="pt", padding=True,
                     truncation=True, max_length=64).to(device)
    out = base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
    pooled = (out * mask).sum(1) / mask.sum(1)
    pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
    sims = pooled @ pooled.T
    n = len(PROBE_SENTENCES)
    off_diag = sims[~torch.eye(n, dtype=torch.bool, device=device)]
    max_sim = off_diag.max().item()
    print(f"  [collapse check{' ' + tag if tag else ''}] max pairwise cosine sim "
          f"among {n} distinct probe sentences: {max_sim:.4f} "
          f"({'COLLAPSED' if max_sim > 0.999 else 'ok, distinct representations'})")
    return max_sim


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    texts = pd.concat([train_df["post"].astype(str), lb_df["post"].astype(str)]).tolist()
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(texts))
    n_val = int(len(texts) * VAL_FRAC)
    val_texts = [texts[i] for i in idx[:n_val]]
    tr_texts = [texts[i] for i in idx[n_val:]]
    print(f"Combined corpus: {len(texts)} posts -> {len(tr_texts)} train / {len(val_texts)} val (MLM phase)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)

    print("\nBEFORE any training:")
    collapse_check(model, tokenizer, device, tag="pre-training sanity")

    freeze_bottom_layers(model, N_UNFROZEN_LAYERS)

    tr_ds = TextOnlyDataset(tr_texts, tokenizer, MAX_LENGTH)
    va_ds = TextOnlyDataset(val_texts, tokenizer, MAX_LENGTH)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "checkpoints_tmp"),
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    print("\nAFTER training (best checkpoint by val MLM loss loaded):")
    max_sim = collapse_check(model, tokenizer, device, tag="post-training sanity")

    if max_sim > 0.999:
        print("\n*** COLLAPSE DETECTED even with safeguards. NOT saving this backbone. ***")
        return

    final_dir = os.path.join(OUT_DIR, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nCollapse check passed. Saved domain-adapted backbone -> {final_dir}")


if __name__ == "__main__":
    main()
