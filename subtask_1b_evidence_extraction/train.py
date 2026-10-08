"""Subtask 1b: Train BIO evidence extractor with GroupKFold, checkpoints, logging."""

import os, csv, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from config import *
from dataset import load_train, get_tokenizer, EvidenceDataset
from model import EvidenceExtractor
from crf import CRF

# ── Phrase F1 (competition metric) ───────────────────────────────────────────

def normalize(s):
    import re
    s = s.replace("‘","'").replace("’","'").replace("“",'"').replace("”",'"')
    s = s.replace("–","-").replace("—","-")
    s = s.replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")
    return re.sub(r"\s+", " ", s).lower().strip()

def phrase_f1_single(pred_spans, true_spans):
    true_n = [normalize(s) for s in true_spans if normalize(s) != "none"]
    pred_n = [normalize(s) for s in pred_spans if normalize(s) != "none"]
    if not true_n:
        return (1.0, 1.0, 1.0) if not pred_n else (0.0, 0.0, 0.0)
    if not pred_n:
        return 0.0, 0.0, 0.0

    used_true, used_pred = set(), set()
    for i, ps in enumerate(pred_n):
        for j, ts in enumerate(true_n):
            if j in used_true:
                continue
            if ts in ps or ps in ts:
                if len(ps.split()) <= 3 * len(ts.split()):
                    used_pred.add(i)
                    used_true.add(j)
                    break
    prec = len(used_pred) / len(pred_n) if pred_n else 0
    rec = len(used_true) / len(true_n) if true_n else 0
    f1 = 2*prec*rec / (prec+rec) if prec+rec else 0
    return prec, rec, f1


# ── Span extraction from BIO preds ──────────────────────────────────────────

def bio_to_spans(bio_ids, offsets, text):
    """Decode BIO tag ids → text spans with merging."""
    raw = []
    cur_s, cur_e = None, None
    for tag_id, (s, e) in zip(bio_ids, offsets):
        if s == 0 and e == 0:
            continue
        if tag_id == BIO2ID["B-EVD"]:
            if cur_s is not None:
                raw.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        elif tag_id == BIO2ID["I-EVD"] and cur_s is not None:
            cur_e = e
        else:
            if cur_s is not None:
                raw.append((cur_s, cur_e))
                cur_s = None
    if cur_s is not None:
        raw.append((cur_s, cur_e))
    if not raw:
        return ["none"]

    # Merge spans with tiny gaps
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
    return spans[:3] if spans else ["none"]  # keep up to 3


# ── Logging & Checkpointing (same pattern as 1a) ────────────────────────────

LOG_FIELDS = ["fold","epoch","train_loss","val_loss","val_phrase_f1",
              "best_pf1","lr","elapsed"]

def init_log():
    path = os.path.join(OUTPUT_DIR, "training_log.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

def log_row(row):
    with open(os.path.join(OUTPUT_DIR, "training_log.csv"), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)

def save_checkpoint(fold, epoch, model, optimizer, scheduler, scaler, best, is_best, crf=None):
    d = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(d, exist_ok=True)
    state = {"fold":fold,"epoch":epoch,"model":model.state_dict(),
             "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
             "scaler":scaler.state_dict() if scaler else None,"best":best,
             "crf": crf.state_dict() if crf is not None else None}
    torch.save(state, os.path.join(d, f"epoch_{epoch}.pt"))
    if is_best:
        torch.save(state, os.path.join(d, "best.pt"))
    # Prune
    fs = sorted([f for f in os.listdir(d) if f.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]))
    while len(fs) > MAX_CHECKPOINTS - 1:
        os.remove(os.path.join(d, fs.pop(0)))

def load_checkpoint(fold):
    d = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    if not os.path.isdir(d):
        return None
    fs = sorted([f for f in os.listdir(d) if f.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1].split(".")[0]))
    return torch.load(os.path.join(d, fs[-1]), weights_only=False) if fs else None

def fold_completed(fold):
    meta = os.path.join(OUTPUT_DIR, "completed_folds.json")
    if os.path.exists(meta):
        with open(meta) as f:
            return str(fold) in json.load(f)
    return False

def mark_fold_done(fold, metrics):
    meta = os.path.join(OUTPUT_DIR, "completed_folds.json")
    data = json.load(open(meta)) if os.path.exists(meta) else {}
    data[str(fold)] = metrics
    json.dump(data, open(meta, "w"), indent=2)
    # Delete epoch checkpoints — only best.pt is needed after fold completes
    d = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    for f in os.listdir(d):
        if f.startswith("epoch_"):
            os.remove(os.path.join(d, f))


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device, crf=None, clip_params=None):
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

        if crf is not None:
            loss = crf(logits.float(), labels, mask.bool()).mean() / GRAD_ACCUM
        else:
            logits_flat = logits.float().view(-1, NUM_BIO)
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1).bool()
            loss = criterion(logits_flat[mask_flat], labels_flat[mask_flat]) / GRAD_ACCUM

        loss.backward()

        if (steps+1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(clip_params or model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total += loss.item() * GRAD_ACCUM
        steps += 1
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")
    pbar.close()
    return total / max(steps,1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, df_val, tokenizer, device, crf=None):
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

        if crf is not None:
            total_loss += crf(logits.float(), labels, mask.bool()).mean().item()
            bio_preds = crf.decode(logits.float(), mask.bool())  # list of variable-length lists
            bs = len(bio_preds)
        else:
            lf = logits.view(-1, NUM_BIO); lb = labels.view(-1); mf = mask.view(-1).bool()
            total_loss += criterion(lf[mf], lb[mf]).item()
            bio_preds = logits.argmax(-1).cpu().numpy()
            bs = bio_preds.shape[0]

        for i in range(bs):
            row_idx = batch_start + i
            if row_idx >= len(df_val): break
            row = df_val.iloc[row_idx]
            enc = tokenizer(str(row["post"]), max_length=MAX_LENGTH,
                            padding="max_length", truncation=True,
                            return_offsets_mapping=True)
            pred_spans = bio_to_spans(bio_preds[i], enc["offset_mapping"], str(row["post"]))
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
    parser.add_argument("--oversample", type=str, default=None,
                        choices=["gpt", "claude", "llama", "all_sources", "verified", "all"],
                        help="Oversample source: gpt | claude | llama | all_sources | verified | all")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override OUTPUT_DIR so a second oversample-source variant "
                             "doesn't clobber the currently-deployed fold checkpoints.")
    parser.add_argument("--use-crf", action="store_true",
                        help="Replace per-token weighted-CE with a linear-chain CRF "
                             "(see crf.py) -- learns valid BIO tag transitions jointly "
                             "with emissions instead of relying only on post-hoc span "
                             "merging. Untried this project; stackable with --oversample.")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Override SEED -- GroupKFold splits are deterministic on "
                             "anon_user_id regardless of seed, so this only changes model "
                             "init/dropout/data order, for seed-bagging ensemble diversity.")
    parser.add_argument("--denoise-indicators", action="store_true",
                        help="Zero evidence_list for indicator-labeled posts during TRAINING "
                             "only (validation labels stay original, for fair evaluation). Fixes "
                             "23/611 indicator posts in train.xlsx that have non-empty evidence "
                             "annotations despite the task's own definition (indicator -> always "
                             "'none') -- label noise that teaches the tagger to hallucinate spans "
                             "on indicator posts. Confirmed elsewhere: doesn't help much alone "
                             "(indicator suppression at inference already forces 'none'), but "
                             "produces a genuinely different model worth having as an ensemble "
                             "member -- same discipline as CRF/seed diversity.")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    init_log()

    tokenizer = get_tokenizer()
    df = load_train(oversample_source=args.oversample)
    print(f"Dataset: {len(df)} | Oversample: {args.oversample or 'none'}")

    # Split synthetic rows — they go to training only, never validation
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df  = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)

    gkf = GroupKFold(n_splits=args.folds)
    groups = orig_df["anon_user_id"].values
    bio_w = torch.tensor(BIO_WEIGHTS, dtype=torch.float32).to(device)

    all_metrics = []
    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        if fold_completed(fold):
            print(f"\nFold {fold} done, skipping.")
            with open(os.path.join(OUTPUT_DIR,"completed_folds.json")) as f:
                all_metrics.append(json.load(f)[str(fold)])
            continue

        tr_df = pd.concat([orig_df.iloc[tr_i], syn_df], ignore_index=True)
        va_df = orig_df.iloc[va_i]

        if args.denoise_indicators:
            tr_df = tr_df.copy()
            is_ind = tr_df["risk_norm"] == "indicator"
            n_denoised = int((is_ind & tr_df["evidence_list"].apply(len).gt(0)).sum())
            tr_df.loc[is_ind, "evidence_list"] = tr_df.loc[is_ind, "evidence_list"].apply(lambda _: [])
            print(f"  --denoise-indicators: zeroed evidence on {n_denoised} indicator posts (train only)")

        print(f"\n{'='*60}\n  FOLD {fold+1}/{args.folds}  "
              f"Train={len(tr_df)} Val={len(va_df)}\n{'='*60}")

        tr_loader = DataLoader(EvidenceDataset(tr_df, tokenizer), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=2, pin_memory=True)
        va_loader = DataLoader(EvidenceDataset(va_df, tokenizer), batch_size=BATCH_SIZE*2,
                               shuffle=False, num_workers=2, pin_memory=True)

        model = EvidenceExtractor().to(device).float()
        model.backbone.gradient_checkpointing_enable()
        criterion = torch.nn.CrossEntropyLoss(weight=bio_w)
        crf = CRF(NUM_BIO).to(device) if args.use_crf else None
        trainable_params = list(model.parameters()) + (list(crf.parameters()) if crf else [])
        optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(tr_loader) * args.epochs // GRAD_ACCUM
        scheduler = get_linear_schedule_with_warmup(optimizer,
                    int(total_steps*WARMUP_RATIO), total_steps)

        start_epoch, best, patience_counter = 0, 0, 0
        ckpt = load_checkpoint(fold)
        if ckpt:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if crf is not None and ckpt.get("crf") is not None:
                crf.load_state_dict(ckpt["crf"])
            start_epoch = ckpt["epoch"]+1; best = ckpt["best"]
            print(f"  Resumed epoch {start_epoch}, best_pf1={best:.4f}")

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            t_loss = train_epoch(model, tr_loader, optimizer, scheduler, criterion, device,
                                  crf=crf, clip_params=trainable_params)
            val = eval_epoch(model, va_loader, criterion, va_df, tokenizer, device, crf=crf)
            elapsed = time.time() - t0
            is_best = val["phrase_f1"] > best
            if is_best:
                best = val["phrase_f1"]
                patience_counter = 0
            else:
                patience_counter += 1
            lr = scheduler.get_last_lr()[0]

            print(f"  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}")
            print(f"    Train loss={t_loss:.4f}  Val loss={val['loss']:.4f}  "
                  f"PhraseF1={val['phrase_f1']:.4f}{'  **BEST**' if is_best else ''}")

            log_row({"fold":fold,"epoch":epoch+1,"train_loss":f"{t_loss:.4f}",
                     "val_loss":f"{val['loss']:.4f}","val_phrase_f1":f"{val['phrase_f1']:.4f}",
                     "best_pf1":f"{best:.4f}","lr":f"{lr:.2e}","elapsed":f"{elapsed:.0f}"})
            save_checkpoint(fold, epoch, model, optimizer, scheduler, None, best, is_best, crf=crf)

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        metrics = {"phrase_f1": best}
        mark_fold_done(fold, metrics)
        all_metrics.append(metrics)

    print(f"\n{'='*60}\n  CV RESULTS\n{'='*60}")
    vals = [m["phrase_f1"] for m in all_metrics]
    print(f"  Phrase F1: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    json.dump({"mean_pf1": float(np.mean(vals)), "folds": all_metrics},
              open(os.path.join(OUTPUT_DIR,"cv_results.json"),"w"), indent=2)


if __name__ == "__main__":
    main()
