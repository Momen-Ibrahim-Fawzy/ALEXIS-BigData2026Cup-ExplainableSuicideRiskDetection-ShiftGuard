"""Domain-adversarial training (DANN, Ganin & Lempitsky 2015) for Sub1a --
the one mechanism this session hasn't tried that's neither post-hoc
(calibration correction, already tried and failed to transfer) nor
task-unaware (MLM continued pretraining, already tried and came back flat).

This directly optimizes for domain-invariant features AS a training
objective: a gradient-reversal layer sits between the shared backbone and a
small domain classifier (train vs. leaderboard, unsupervised -- no labels
needed for leaderboard.xlsx, just which file a post came from). The domain
classifier tries to tell train and leaderboard posts apart; gradient
reversal means the BACKBONE is simultaneously trained to make that
increasingly impossible, directly targeting the measured AUC~0.66
adversarial-validation shift as part of the training loop itself, not as a
diagnostic or a post-hoc correction.

Deliberately a SINGLE, non-searched attempt: one lambda schedule (the
standard DANN ramp-up, Ganin & Lempitsky's own recommended formula), one
domain-loss weight, no sweep. Given this session's own finding that
search-selected candidates have 2-3x larger local-to-real gaps than single
principled changes, doing anything else here would undermine the point of
trying this in the first place.

Usage:
    python train_dann_domain_adversarial.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import TextDataset, RISK2ID, DATA_DIR, RESULTS_DIR

MODEL_NAME = "microsoft/deberta-v3-large"
EXP_NAME = "exp48_dann_domain_adversarial"
OUT_DIR = os.path.join(RESULTS_DIR, EXP_NAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
N_FOLDS = 5
SEED = 42

LR = 2e-5
EPOCHS = 32
PATIENCE = 4
BATCH_SIZE = 8
LABEL_SMOOTHING = 0.05
CLASS_WEIGHTS = [0.669, 0.788, 1.045, 3.586]
DOMAIN_LOSS_WEIGHT = 1.0   # standard DANN default; the GRL's own lambda ramp already
                           # modulates the effective strength, so this stays at 1.0


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DANNClassifier(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.class_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, num_classes))
        self.domain_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, 2)
        )

    def pool(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, lambd=0.0):
        pooled = self.pool(input_ids, attention_mask)
        class_logits = self.class_head(pooled)
        domain_logits = self.domain_head(grad_reverse(pooled, lambd))
        return class_logits, domain_logits


class UnlabeledTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(str(self.texts[i]), max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels_all = [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            class_logits, _ = model(ids, mask, lambd=0.0)
        preds.append(class_logits.float().argmax(-1).cpu().numpy())
        labels_all.append(batch["labels"].numpy())
    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    return f1_score(labels_all, preds, average="weighted", zero_division=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    groups = df["anon_user_id"].values

    lb_df = pd.read_excel(os.path.join(DATA_DIR, "leaderboard.xlsx"), engine="openpyxl")
    print(f"train.xlsx: {len(df)} rows, leaderboard.xlsx (unlabeled, domain target): {len(lb_df)} rows")

    gkf = GroupKFold(n_splits=N_FOLDS)
    class_w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    fold_wf1s = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n{'#'*70}\n  FOLD {fold+1}/{N_FOLDS}\n{'#'*70}")
        tr_df, va_df = df.iloc[tr_i], df.iloc[va_i]

        tr_ds = TextDataset(tr_df["post"].tolist(), labels[tr_i].tolist(), tokenizer, MAX_LENGTH)
        va_ds = TextDataset(va_df["post"].tolist(), labels[va_i].tolist(), tokenizer, MAX_LENGTH)
        lb_ds = UnlabeledTextDataset(lb_df["post"].tolist(), tokenizer, MAX_LENGTH)

        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
        va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
        lb_ld_cycle = cycle(DataLoader(lb_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True))

        torch.manual_seed(SEED + fold)
        model = DANNClassifier(MODEL_NAME).to(DEVICE).float()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps_per_epoch = len(tr_ld)
        total_steps = steps_per_epoch * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
        class_criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING)
        domain_criterion = nn.CrossEntropyLoss()

        best_wf1, best_state, patience_ctr, global_step = 0, None, 0, 0
        for epoch in range(EPOCHS):
            model.train()
            for tr_batch in tr_ld:
                p = global_step / total_steps
                lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Ganin & Lempitsky ramp-up

                lb_batch = next(lb_ld_cycle)
                ids = torch.cat([tr_batch["input_ids"], lb_batch["input_ids"]], dim=0).to(DEVICE)
                mask = torch.cat([tr_batch["attention_mask"], lb_batch["attention_mask"]], dim=0).to(DEVICE)
                class_labels = tr_batch["labels"].to(DEVICE)
                domain_labels = torch.cat([
                    torch.zeros(len(tr_batch["input_ids"]), dtype=torch.long),
                    torch.ones(len(lb_batch["input_ids"]), dtype=torch.long),
                ]).to(DEVICE)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    class_logits, domain_logits = model(ids, mask, lambd=lambd)
                    class_loss = class_criterion(class_logits[:len(class_labels)].float(), class_labels)
                    domain_loss = domain_criterion(domain_logits.float(), domain_labels)
                    loss = class_loss + DOMAIN_LOSS_WEIGHT * domain_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

            val_wf1 = evaluate(model, va_ld)
            print(f"  epoch {epoch+1}: lambd={lambd:.3f} class_loss={class_loss.item():.4f} "
                  f"domain_loss={domain_loss.item():.4f} val_wf1={val_wf1:.4f}", flush=True)
            if val_wf1 > best_wf1:
                best_wf1, patience_ctr = val_wf1, 0
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"  early stop at epoch {epoch+1}")
                    break

        # save only the parts needed for downstream inference (backbone + class_head),
        # matching Classifier's state_dict key naming so it can be reloaded interchangeably
        full_state = best_state
        classifier_compatible_state = {k: v for k, v in full_state.items()
                                        if k.startswith("backbone.") or k.startswith("class_head.")}
        classifier_compatible_state = {
            (k.replace("class_head.", "head.") if k.startswith("class_head.") else k): v
            for k, v in classifier_compatible_state.items()
        }
        torch.save(classifier_compatible_state, os.path.join(OUT_DIR, f"best_fold{fold}.pt"))
        print(f"  fold {fold} best val WF1: {best_wf1:.4f}")
        fold_wf1s.append(best_wf1)
        del model
        torch.cuda.empty_cache()

    mean_wf1 = float(np.mean(fold_wf1s))
    std_wf1 = float(np.std(fold_wf1s))
    print(f"\n{'='*70}\n  {EXP_NAME}: WF1 = {mean_wf1:.4f} +/- {std_wf1:.4f}\n"
          f"  (vs exp09's 0.8003+/-0.0241)\n{'='*70}")
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"experiment": EXP_NAME, "mean_wf1": mean_wf1, "std_wf1": std_wf1,
                    "fold_wf1": fold_wf1s}, f, indent=2)


if __name__ == "__main__":
    main()
