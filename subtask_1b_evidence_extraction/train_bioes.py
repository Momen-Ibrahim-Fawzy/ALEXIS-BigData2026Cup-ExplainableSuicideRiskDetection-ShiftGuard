"""Subtask 1b: BIOES tagging retry (O/B-EVD/I-EVD/E-EVD/S-EVD instead of plain
BIO's O/B-EVD/I-EVD).

Motivation (from web research into external ideas for this project): BIOES
adds explicit End and Single-token tags, giving the model a direct signal for
span *boundaries* rather than just "inside vs outside" -- IO/BIO schemes tend
toward higher recall, BIOES toward higher precision on span boundaries. This
project's evidence spans are short (often 1-5 words), so boundary precision
plausibly matters more here than it would for longer spans.

Self-contained: reuses the existing model.py (EvidenceExtractor), crf.py (CRF,
already generic over NUM_BIO -- no changes needed for 5 classes instead of 3),
and train.py's phrase_f1_single/normalize (the eval metric itself is
tag-scheme-independent, only the span DECODER differs). Only the label-
building (BIO2ID with 5 tags, and knowing which token is last in a span) and
the span decoder are new -- kept in this file rather than touching the
validated dataset.py/train.py, so the existing BIO family checkpoints and
composition search are completely unaffected.

Trains the "denoised" recipe (denoise_indicators=True) for a clean, direct
comparison against the existing outputs_denoised family's true-OOF Phrase F1.

Usage:
    python train_bioes.py --output-dir outputs_bioes_denoised --denoise-indicators
"""

import os, csv, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from config import *
from dataset import load_train
from model import EvidenceExtractor
from crf import CRF
from train import normalize, phrase_f1_single

BIOES_TAGS = ["O", "B-EVD", "I-EVD", "E-EVD", "S-EVD"]
BIOES2ID = {t: i for i, t in enumerate(BIOES_TAGS)}
ID2BIOES = {i: t for i, t in enumerate(BIOES_TAGS)}
NUM_BIOES = len(BIOES_TAGS)
BIOES_WEIGHTS = [0.1, 8.0, 8.0, 8.0, 8.0]


class EvidenceDatasetBIOES(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["post"])

        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt",
                             return_offsets_mapping=True)
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        offsets = enc["offset_mapping"].squeeze(0)

        if self.is_test:
            item["offset_mapping"] = offsets
            return item

        bio = torch.zeros(self.max_length, dtype=torch.long)  # all O
        evidence_list = row.get("evidence_list", [])

        if evidence_list:
            text_lower = text.lower()
            for span_text in evidence_list:
                span_lower = span_text.lower().strip()
                if not span_lower or span_lower == "none":
                    continue
                start_char = text_lower.find(span_lower)
                if start_char == -1:
                    continue
                end_char = start_char + len(span_lower)

                # Collect all token indices covered by this span first, so we
                # can tell which is first/last for correct BIOES assignment.
                tok_idxs = []
                for tok_i in range(self.max_length):
                    ts, te = offsets[tok_i].tolist()
                    if ts == 0 and te == 0:
                        continue
                    if te <= start_char:
                        continue
                    if ts >= end_char:
                        break
                    tok_idxs.append(tok_i)

                if not tok_idxs:
                    continue
                if len(tok_idxs) == 1:
                    bio[tok_idxs[0]] = BIOES2ID["S-EVD"]
                else:
                    bio[tok_idxs[0]] = BIOES2ID["B-EVD"]
                    bio[tok_idxs[-1]] = BIOES2ID["E-EVD"]
                    for mid in tok_idxs[1:-1]:
                        bio[mid] = BIOES2ID["I-EVD"]

        item["labels"] = bio
        return item


def bioes_to_spans(bioes_ids, offsets, text):
    """Decode BIOES tag ids -> text spans. Lenient about malformed sequences
    (e.g. a B not followed by an E before the next O/B/S) -- closes the span
    at the last extended position instead of discarding it, same tolerance
    train.py's plain-BIO decoder already has for imperfect model output."""
    raw = []
    cur_s, cur_e = None, None
    for tag_id, (s, e) in zip(bioes_ids, offsets):
        if s == 0 and e == 0:
            continue
        tag = ID2BIOES[tag_id]
        if tag == "S-EVD":
            if cur_s is not None:
                raw.append((cur_s, cur_e))
                cur_s = None
            raw.append((s, e))
        elif tag == "B-EVD":
            if cur_s is not None:
                raw.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        elif tag == "I-EVD":
            if cur_s is not None:
                cur_e = e
        elif tag == "E-EVD":
            if cur_s is not None:
                cur_e = e
                raw.append((cur_s, cur_e))
                cur_s = None
            else:
                raw.append((s, e))  # E with no preceding B -- treat as a lone span
        else:  # O
            if cur_s is not None:
                raw.append((cur_s, cur_e))
                cur_s = None
    if cur_s is not None:
        raw.append((cur_s, cur_e))
    if not raw:
        return ["none"]

    raw.sort(key=lambda x: x[0])
    merged = [raw[0]]
    for s, e in raw[1:]:
        ps, pe = merged[-1]
        gap_text = text[pe:s].strip()
        if s - pe <= 5 and len(gap_text) <= 3:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    spans = []
    for s, e in merged:
        t = text[s:e].strip()
        if len(t) > 2 and t.lower() not in {"i","me","my","he","she","we","im","i'm"}:
            spans.append(t)
    return spans[:3] if spans else ["none"]


# -- Checkpointing (identical scheme to train.py) ------------------------------

def init_log(output_dir):
    path = os.path.join(output_dir, "training_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["fold","epoch","train_loss","val_loss",
                           "val_phrase_f1","best_pf1","lr","elapsed"]).writeheader()

def log_row(output_dir, row):
    with open(os.path.join(output_dir, "training_log.csv"), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=["fold","epoch","train_loss","val_loss",
                       "val_phrase_f1","best_pf1","lr","elapsed"]).writerow(row)

def save_checkpoint(output_dir, fold, epoch, model, optimizer, scheduler, best, is_best, crf=None):
    d = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(d, exist_ok=True)
    state = {"fold":fold,"epoch":epoch,"model":model.state_dict(),
             "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
             "best":best,"crf": crf.state_dict() if crf is not None else None}
    torch.save(state, os.path.join(d, f"epoch_{epoch}.pt"))
    if is_best:
        torch.save(state, os.path.join(d, "best.pt"))
    fs = sorted([f for f in os.listdir(d) if f.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]))
    while len(fs) > MAX_CHECKPOINTS - 1:
        os.remove(os.path.join(d, fs.pop(0)))

def load_checkpoint(output_dir, fold):
    d = os.path.join(output_dir, f"fold_{fold}")
    if not os.path.isdir(d):
        return None
    fs = sorted([f for f in os.listdir(d) if f.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]))
    return torch.load(os.path.join(d, fs[-1]), weights_only=False) if fs else None

def fold_completed(output_dir, fold):
    meta = os.path.join(output_dir, "completed_folds.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return str(fold) in json.load(f)
    return False

def mark_fold_done(output_dir, fold, metrics):
    meta = os.path.join(output_dir, "completed_folds.json")
    data = json.load(open(meta)) if os.path.exists(meta) else {}
    data[str(fold)] = metrics
    json.dump(data, open(meta, "w"), indent=2)
    d = os.path.join(output_dir, f"fold_{fold}")
    for f in os.listdir(d):
        if f.startswith("epoch_"):
            os.remove(os.path.join(d, f))


def train_epoch(model, loader, optimizer, scheduler, criterion, device, crf, clip_params):
    model.train()
    total, steps = 0, 0
    use_amp = device.type == "cuda"
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(ids, mask, ttype)

        loss = crf(logits.float(), labels, mask.bool()).mean() / GRAD_ACCUM
        loss.backward()

        if (steps+1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total += loss.item() * GRAD_ACCUM
        steps += 1
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")
    pbar.close()
    return total / max(steps,1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, df_val, tokenizer, device, crf):
    model.eval()
    total_loss, all_pf1 = 0, []
    batch_start = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        ttype = batch.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(device)

        logits = model(ids, mask, ttype)
        total_loss += crf(logits.float(), labels, mask.bool()).mean().item()
        bio_preds = crf.decode(logits.float(), mask.bool())
        bs = len(bio_preds)

        for i in range(bs):
            row_idx = batch_start + i
            if row_idx >= len(df_val): break
            row = df_val.iloc[row_idx]
            enc = tokenizer(str(row["post"]), max_length=MAX_LENGTH,
                            padding="max_length", truncation=True,
                            return_offsets_mapping=True)
            pred_spans = bioes_to_spans(bio_preds[i], enc["offset_mapping"], str(row["post"]))
            true_spans = row.get("evidence_list", []) or ["none"]
            _, _, f1 = phrase_f1_single(pred_spans, true_spans)
            all_pf1.append(f1)

        batch_start += bs

    n = max(len(all_pf1),1)
    return {"loss": total_loss/n, "phrase_f1": np.mean(all_pf1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--denoise-indicators", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Tags: {BIOES_TAGS}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    init_log(output_dir)

    from dataset import get_tokenizer
    tokenizer = get_tokenizer()
    df = load_train(oversample_source=None)
    print(f"Dataset: {len(df)}")

    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df  = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=args.folds)
    groups = orig_df["anon_user_id"].values
    bio_w = torch.tensor(BIOES_WEIGHTS, dtype=torch.float32).to(device)

    all_metrics = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        if fold_completed(output_dir, fold):
            print(f"\nFold {fold} done, skipping.")
            with open(os.path.join(output_dir,"completed_folds.json")) as f:
                all_metrics.append(json.load(f)[str(fold)])
            continue

        tr_df = pd.concat([orig_df.iloc[tr_i], syn_df], ignore_index=True)
        va_df = orig_df.iloc[va_i]

        if args.denoise_indicators:
            tr_df = tr_df.copy()
            is_ind = tr_df["risk_norm"] == "indicator"
            tr_df.loc[is_ind, "evidence_list"] = tr_df.loc[is_ind, "evidence_list"].apply(lambda _: [])

        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  Train={len(tr_df)} Val={len(va_df)}\n{'='*60}")

        tr_loader = DataLoader(EvidenceDatasetBIOES(tr_df, tokenizer), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(EvidenceDatasetBIOES(va_df, tokenizer), batch_size=BATCH_SIZE*2,
                               shuffle=False, num_workers=2, pin_memory=True)

        model = EvidenceExtractor(model_name=MODEL_NAME).to(device).float()
        # NUM_BIO is imported via `from config import *` -- override the head size for BIOES
        model.head = torch.nn.Linear(model.backbone.config.hidden_size, NUM_BIOES).to(device)
        model.backbone.gradient_checkpointing_enable()
        criterion = torch.nn.CrossEntropyLoss(weight=bio_w)
        crf = CRF(NUM_BIOES).to(device)
        trainable_params = list(model.parameters()) + list(crf.parameters())
        optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_loader) * args.epochs // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*WARMUP_RATIO), total_steps)

        start_epoch, best, patience_counter = 0, 0, 0
        ckpt = load_checkpoint(output_dir, fold)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            crf.load_state_dict(ckpt["crf"])
            start_epoch = ckpt["epoch"]+1; best = ckpt["best"]
            print(f"  Resumed epoch {start_epoch}, best_pf1={best:.4f}")

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            t_loss = train_epoch(model, tr_loader, optimizer, scheduler, criterion, device,
                                  crf, trainable_params)
            val = eval_epoch(model, va_loader, criterion, va_df, tokenizer, device, crf)
            elapsed = time.time() - t0
            is_best = val["phrase_f1"] > best
            if is_best:
                best = val["phrase_f1"]; patience_counter = 0
            else:
                patience_counter += 1
            lr = scheduler.get_last_lr()[0]

            print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}")
            print(f"    Train loss={t_loss:.4f}  Val loss={val['loss']:.4f}  "
                  f"PhraseF1={val['phrase_f1']:.4f}{'  **BEST**' if is_best else ''}")

            log_row(output_dir, {"fold":fold,"epoch":epoch+1,"train_loss":f"{t_loss:.4f}",
                     "val_loss":f"{val['loss']:.4f}","val_phrase_f1":f"{val['phrase_f1']:.4f}",
                     "best_pf1":f"{best:.4f}","lr":f"{lr:.2e}","elapsed":f"{elapsed:.0f}"})
            save_checkpoint(output_dir, fold, epoch, model, optimizer, scheduler, best, is_best, crf)

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        metrics = {"phrase_f1": best}
        mark_fold_done(output_dir, fold, metrics)
        all_metrics.append(metrics)
        del model, crf, optimizer, scheduler
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n  CV RESULTS\n{'='*60}")
    vals = [m["phrase_f1"] for m in all_metrics]
    print(f"  Phrase F1: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(f"  (compare vs outputs_denoised/crf_denoised family's own true-OOF Phrase F1)")

    json.dump({"mean_pf1": float(np.mean(vals)), "folds": all_metrics, "tag_scheme": "BIOES"},
              open(os.path.join(output_dir,"cv_results.json"),"w"), indent=2)


if __name__ == "__main__":
    main()
