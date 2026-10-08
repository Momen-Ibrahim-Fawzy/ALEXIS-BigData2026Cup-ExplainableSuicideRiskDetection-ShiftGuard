"""Unified experiment trainer for Subtask 1a.

Usage:
    python trainer.py exp01_deberta_large         # run one experiment
    python trainer.py --dry-run                    # pipeline check (fast)
    python trainer.py --all                        # run all experiments
    python trainer.py exp01_deberta_large exp03_roberta_large  # run multiple
"""

import os, sys, csv, json, time, argparse, shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
)
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import ast

from experiment_config import EXPERIMENTS, DRY_RUN

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RISK2ID = {r: i for i, r in enumerate(RISK_LEVELS)}
ID2RISK = {i: r for i, r in enumerate(RISK_LEVELS)}
SEED = 42


# ── FGM adversarial training ────────────────────────────────────────────────
# Standard Miyato et al. Fast Gradient Method: after a clean backward pass,
# perturb the word-embedding weights along the (normalized) loss-gradient
# direction, take a second forward/backward pass on the perturbed embeddings
# (gradients accumulate on top of the clean-pass gradients), then restore the
# embeddings before the optimizer step.

class FGM:
    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ── Supervised Contrastive Loss (Khosla et al. 2020) ────────────────────────
# Pulls same-class pooled embeddings together, pushes different-class ones
# apart, within each batch. Combined with CE: total = CE + alpha * SupCon.

def supcon_loss(embeddings, labels, temperature=0.07):
    embeddings = nn.functional.normalize(embeddings, dim=1)
    B = embeddings.size(0)
    sim = torch.matmul(embeddings, embeddings.T) / temperature  # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    sim = sim.masked_fill(self_mask, -1e9)  # exclude self from both pos and denom

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask  # (B, B)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = pos_mask.sum(1)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1) / pos_counts.clamp(min=1)

    valid = pos_counts > 0  # anchors with at least one same-class partner in-batch
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return -mean_log_prob_pos[valid].mean()


# ── AWP: Adversarial Weight Perturbation (Wu et al. 2020) ───────────────────
# Perturbs the backbone's own WEIGHTS (not input embeddings, unlike FGM),
# scaled relative to each weight tensor's own norm and clipped to a relative
# epsilon ball, restricted to backbone linear/attention weights (excludes
# embeddings and LayerNorm, standard AWP scope).

class AWP:
    def __init__(self, model, adv_lr=1.0, adv_eps=0.01):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def _target(self, name, param):
        return (param.requires_grad and param.grad is not None and "weight" in name
                and "LayerNorm" not in name and "embeddings" not in name
                and "backbone" in name)

    def attack(self):
        for name, param in self.model.named_parameters():
            if not self._target(name, param):
                continue
            norm_grad = torch.norm(param.grad)
            norm_data = torch.norm(param.data)
            if norm_grad == 0 or torch.isnan(norm_grad):
                continue
            if name not in self.backup:
                self.backup[name] = param.data.clone()
            r = self.adv_lr * param.grad / (norm_grad + 1e-6) * (norm_data + 1e-6)
            param.data.add_(r)
            # clip to a relative epsilon ball around the pre-attack weights
            lo = self.backup[name] - self.adv_eps * (norm_data + 1e-6)
            hi = self.backup[name] + self.adv_eps * (norm_data + 1e-6)
            param.data.copy_(torch.max(torch.min(param.data, hi), lo))

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ── FreeLB (Zhu et al. 2020) ─────────────────────────────────────────────────
# Multi-step embedding-space adversarial training: K ascent steps accumulate
# gradient across K adversarial "views" of the same batch (vs. FGM's single
# step), simulating a larger effective batch while training against a
# stronger adversary.

class FreeLB:
    def __init__(self, adv_k=2, adv_lr=1e-2, adv_eps=1e-1):
        self.adv_k = adv_k
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps

    def attack_and_backward(self, model, ids, mask, labels, sw, criterion, use_amp, grad_accum):
        """Runs K adversarial ascent steps in embedding space, backward()-ing at each
        step (gradients accumulate on top of whatever's already in .grad). Returns the
        last step's loss value for logging."""
        embeds = model.embed_inputs(ids).detach()
        delta = torch.zeros_like(embeds).uniform_(-self.adv_eps, self.adv_eps)
        delta.requires_grad_()

        last_loss = None
        for step in range(self.adv_k):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inputs_embeds=embeds + delta, attention_mask=mask)
            per_sample = criterion(logits.float(), labels)
            loss = (per_sample * sw).sum() / sw.sum() / grad_accum / self.adv_k
            loss.backward()
            last_loss = loss

            delta_grad = delta.grad.clone()
            norm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
            norm = norm.clamp(min=1e-6)
            delta = (delta + self.adv_lr * delta_grad / norm).detach()
            delta = delta.clamp(-self.adv_eps, self.adv_eps)
            delta.requires_grad_()

        return last_loss


# ── Lookahead optimizer (Zhang et al. 2019) + EMA (Polyak averaging) ───────

class Lookahead:
    """Wraps an existing optimizer: every k inner ('fast') steps, interpolates the
    tracked 'slow' weights alpha-of-the-way toward the fast weights, then resets the
    fast weights to the slow weights."""
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.counter = 0
        self.slow_state = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                self.slow_state[p] = p.data.clone()

    def step(self):
        self.optimizer.step()
        self.counter += 1
        if self.counter >= self.k:
            self.counter = 0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    slow = self.slow_state[p]
                    slow += self.alpha * (p.data - slow)
                    p.data.copy_(slow)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class EMA:
    """Exponential moving average of model weights; evaluate/save using ema.shadow,
    not the raw (noisier) training weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_to(self, model):
        """Returns the original weights so they can be restored after eval."""
        backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


# ── R-Drop (Liang et al. 2021) ───────────────────────────────────────────────
# Two independent forward passes of the SAME input through the SAME model in
# train() mode (dropout active both times -> two different dropout masks),
# consistency-regularized via symmetric KL divergence between the two output
# distributions. General-purpose regularizer, mechanically distinct from the
# adversarial-perturbation family (FGM/AWP/FreeLB) and from SupCon
# (representation-space contrastive) -- doesn't perturb inputs or weights at
# all, just penalizes the model for being sensitive to its own dropout noise.

def rdrop_kl(logits1, logits2):
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    kl1 = F.kl_div(p1, p2, reduction="batchmean", log_target=True)
    kl2 = F.kl_div(p2, p1, reduction="batchmean", log_target=True)
    return (kl1 + kl2) / 2


# ── Multi-sample dropout (Inoue 2019) ────────────────────────────────────────
# One backbone forward pass (expensive), but K independent dropout masks applied
# to the resulting pooled representation before the classification head (cheap --
# the head is tiny), averaging the K per-sample losses. Cheaper than R-Drop
# (which reruns the whole backbone) since only the small head is repeated.

def multi_sample_dropout_loss(model, pooled, labels, sw, criterion, k=4):
    losses = []
    all_logits = []
    for _ in range(k):
        logits_i = model.head(pooled)
        all_logits.append(logits_i)
        per_sample = criterion(logits_i.float(), labels)
        losses.append((per_sample * sw).sum() / sw.sum())
    return sum(losses) / k, torch.stack(all_logits, dim=0).mean(dim=0)


# ── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha else None
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal = alpha[targets] * focal
        return focal.mean()


# ── Macro Double Soft-F1 Loss (adapted for 4-way softmax) ───────────────────
# The IEEE BigData 2024 Cup winning suicide-detection approach used a custom
# "macro_double_soft_f1" loss instead of any cross-entropy variant -- directly
# optimizing a differentiable approximation of F1 rather than a proxy loss.
# Every loss tried in this project so far (weighted CE, label smoothing, focal,
# the CORAL blend) is a CE-family proxy; this is a mechanistically different
# training objective.
#
# The reference implementation is for multi-label sigmoid outputs (independent
# per-label binary decisions); adapted here for single-label 4-way softmax by
# treating each class's softmax probability as that class's "positive"
# probability and (1 - p) as its "negative" probability, building soft
# TP/FP/FN/TN per class exactly as in the reference, then averaging.
#
# One deliberate change from the reference "macro" version: this project's
# actual eval metric is WEIGHTED F1, not macro F1 (confirmed from real
# submission numbers: 0.7*Subtask1 + 0.3*Subtask2 matches exactly). So the
# per-class costs are combined with the same class weights exp09 already uses
# for weighted CE, rather than a plain unweighted mean across classes --
# aligning the loss with the actual target metric rather than blindly copying
# the reference's "macro" aggregation.

def macro_double_soft_f1_loss(logits, labels, num_classes, class_weights=None, eps=1e-16):
    probs = torch.softmax(logits, dim=-1)                          # (B, C)
    y_onehot = F.one_hot(labels, num_classes).float()              # (B, C)

    tp = (probs * y_onehot).sum(dim=0)                             # (C,)
    fp = (probs * (1 - y_onehot)).sum(dim=0)
    fn = ((1 - probs) * y_onehot).sum(dim=0)
    tn = ((1 - probs) * (1 - y_onehot)).sum(dim=0)

    soft_f1_pos = 2 * tp / (2 * tp + fn + fp + eps)
    soft_f1_neg = 2 * tn / (2 * tn + fn + fp + eps)
    cost = 1 - 0.5 * (soft_f1_pos + soft_f1_neg)                   # (C,) per-class cost

    if class_weights is not None:
        return (cost * class_weights).sum() / class_weights.sum()
    return cost.mean()


class MacroDoubleSoftF1Loss(nn.Module):
    def __init__(self, num_classes=4, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

    def forward(self, logits, targets):
        cw = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return macro_double_soft_f1_loss(logits, targets, self.num_classes, cw)


# ── Class Distance Weighted Cross-Entropy (CDW-CE) ───────────────────────────
# From Polat et al. 2022 (arXiv 2202.05167) via its reference implementation
# (github.com/carlomarxdk/cdw-cross-entropy-loss). An ordinal loss, but a
# structurally different mechanism from CORAL (already tried in this project,
# 0.7958 solo -- a near-miss vs exp09's 0.8003): CORAL adds an auxiliary
# cumulative-threshold head; CDW-CE changes nothing about the model, only the
# loss. For each sample, it penalizes probability mass placed on WRONG classes
# in proportion to (ordinal distance)^alpha -- attempt predicted for a true
# indicator (distance 3) is penalized far more than ideation predicted for a
# true indicator (distance 1). Notably it never directly rewards the true
# class's own probability the way -log(p_true) does; it works entirely by
# suppressing wrong-class mass, so correct predictions only benefit indirectly
# through the softmax simplex constraint (probabilities sum to 1).
#
# Reference is unweighted by class frequency. Extended here with this
# project's standard per-class weights (matching weighted_ce's convention,
# used everywhere else to handle the severe indicator/attempt imbalance) by
# scaling each sample's loss by its true class's weight -- the ordinal
# distance-weighting mechanism itself is otherwise unchanged from the
# reference implementation.

class CDWCELoss(nn.Module):
    def __init__(self, num_classes=4, alpha=2.0, class_weights=None, eps=1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.eps = eps
        self.register_buffer("class_idx", torch.arange(num_classes, dtype=torch.float32))
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=-1)                          # (B, C)
        dist = torch.abs(self.class_idx.to(logits.device) - targets.view(-1, 1).float())  # (B, C)
        weights = torch.pow(dist, self.alpha)

        per_sample = -torch.sum(weights * torch.log(1 - probs + self.eps), dim=-1)  # (B,)

        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)[targets]
            per_sample = per_sample * cw
        return per_sample.mean()


# ── Model ────────────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, num_classes))

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                return_pooled=False, **kwargs):
        if inputs_embeds is not None:
            out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens — works for all architectures
        # (CLS pooling [:, 0] breaks XLNet which places CLS at the last position)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.head(pooled)
        if return_pooled:
            return logits, pooled
        return logits

    def embed_inputs(self, input_ids):
        """Word-embedding lookup only, for FreeLB-style embedding-space perturbation."""
        return self.backbone.embeddings.word_embeddings(input_ids)


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_data(data_mode, pseudo_conf_threshold=0.75, pseudo_weight=0.7):
    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df["sample_weight"] = 1.0

    # Data mode → oversample file mapping
    ov_files = {
        "oversample_all":          "oversample.json",          # GPT-OSS only (809)
        "oversample_verified":     "oversample.json",          # GPT-OSS verified only
        "oversample_gpt":          "oversample_gpt_oss.json",  # GPT-OSS (809)
        "oversample_claude":       "oversample_claude.json",   # Claude (809)
        "oversample_llama":        "oversample_llama.json",    # LLAMA (809)
        "oversample_all_sources":  "oversample_all_sources.json",  # All 3 sources (2427)
        "oversample_contrastive": "oversample_contrastive.json",  # hand-authored minimal pairs (60)
        "oversample_contrastive_v2": "oversample_contrastive_v2.json",  # farewell-note + method-specificity pairs (34)
    }

    if data_mode in ov_files:
        ov_path = os.path.join(DATA_DIR, ov_files[data_mode])
        with open(ov_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        rows = []
        for s in samples:
            if data_mode == "oversample_verified" and not s.get("both_agree", True):
                continue
            rows.append({"row_id": s.get("id",""), "anon_user_id": "SYN",
                         "post": s["text"], "risk_norm": s["risk_level"], "sample_weight": 1.0})
        if rows:
            df = pd.concat([df[["row_id","anon_user_id","post","risk_norm","sample_weight"]],
                            pd.DataFrame(rows)], ignore_index=True)
            print(f"  Added {len(rows)} oversampled posts ({data_mode})")

    elif data_mode == "pseudo_leaderboard":
        # Self-training: exp09 5-fold-ensemble pseudo-labels on the 378 unlabeled
        # leaderboard posts (see pseudo_label_leaderboard.py), filtered by average
        # softmax confidence and down-weighted relative to gold labels. Marked
        # anon_user_id="SYN" so it's automatically train-only, same protection the
        # existing oversample sources get -- never enters a validation fold.
        pl_path = os.path.join(DATA_DIR, "pseudo_leaderboard_full.json")
        with open(pl_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        rows = []
        for s in samples:
            if s["confidence"] < pseudo_conf_threshold:
                continue
            rows.append({"row_id": s["row_id"], "anon_user_id": "SYN",
                         "post": s["text"], "risk_norm": s["risk_level"],
                         "sample_weight": pseudo_weight})
        if rows:
            df = pd.concat([df[["row_id","anon_user_id","post","risk_norm","sample_weight"]],
                            pd.DataFrame(rows)], ignore_index=True)
            print(f"  Added {len(rows)} pseudo-labeled leaderboard posts "
                  f"(confidence>={pseudo_conf_threshold}, weight={pseudo_weight})")

    return df


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, weights=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weights = weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(str(self.texts[i]), max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        if self.weights is not None:
            item["sample_weight"] = torch.tensor(self.weights[i], dtype=torch.float32)
        return item


# ── Trainer ──────────────────────────────────────────────────────────────────

def run_experiment(name, cfg):
    exp_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    summary_path = os.path.join(exp_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        print(f"\n  SKIP {name} — already complete (WF1={existing['mean_wf1']:.4f})")
        return existing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {cfg['description']}")
    print(f"  Model: {cfg['model_name']}")
    print(f"  Data: {cfg['data']}  Loss: {cfg['loss']}  LR: {cfg['lr']}")
    print(f"  Epochs: {cfg['epochs']}  Batch: {cfg['batch_size']}  Patience: {cfg['patience']}")
    print(f"  Device: {device}")
    print(f"{'#'*70}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    df = load_data(cfg["data"])
    n_folds = cfg.get("_folds", 5)

    # Separate synthetic rows so they only ever appear in training, never validation.
    # GroupKFold treats all "SYN" rows as one group → without this, they land in
    # exactly one fold's validation set, making cross-experiment comparison invalid.
    orig_df = df[df["anon_user_id"] != "SYN"].reset_index(drop=True)
    syn_df  = df[df["anon_user_id"] == "SYN"].reset_index(drop=True)
    n_syn   = len(syn_df)
    print(f"  Data: {len(orig_df)} original posts"
          + (f" + {n_syn} synthetic (train-only)" if n_syn else "")
          + f", {n_folds} folds")

    # Log file
    log_path = os.path.join(exp_dir, "training_log.csv")
    log_fields = ["fold","epoch","train_loss","val_loss","val_wf1","val_mf1",
                  "val_acc","best_wf1","lr","elapsed"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # Split on original data only — validation is always original, never synthetic
    gkf = GroupKFold(n_splits=n_folds)
    groups = orig_df["anon_user_id"].values
    all_wf1 = []

    for fold, (tr_i, va_i) in enumerate(gkf.split(orig_df, groups=groups)):
        tr_df = pd.concat([orig_df.iloc[tr_i], syn_df], ignore_index=True)
        va_df = orig_df.iloc[va_i]
        print(f"\n  Fold {fold+1}/{n_folds}  Train={len(tr_df)} Val={len(va_df)}")

        tr_ds = TextDataset(tr_df["post"].tolist(), [RISK2ID[r] for r in tr_df["risk_norm"]],
                            tokenizer, cfg["max_length"], weights=tr_df["sample_weight"].tolist())
        va_ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]],
                            tokenizer, cfg["max_length"])
        tr_ld = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=2, pin_memory=True)
        va_ld = DataLoader(va_ds, batch_size=cfg["batch_size"]*2, shuffle=False,
                           num_workers=2, pin_memory=True)

        model = Classifier(cfg["model_name"]).to(device).float()
        model.backbone.gradient_checkpointing_enable()

        fgm = FGM(model, epsilon=cfg.get("fgm_epsilon", 1.0)) if cfg.get("fgm") else None
        awp = AWP(model, adv_lr=cfg.get("awp_lr", 1.0), adv_eps=cfg.get("awp_eps", 0.01)) if cfg.get("awp") else None
        freelb = FreeLB(adv_k=cfg.get("freelb_k", 2), adv_lr=cfg.get("freelb_lr", 1e-2),
                         adv_eps=cfg.get("freelb_eps", 1e-1)) if cfg.get("freelb") else None
        use_supcon = cfg.get("supcon", False)
        supcon_alpha = cfg.get("supcon_alpha", 0.3)
        use_rdrop = cfg.get("rdrop", False)
        rdrop_alpha = cfg.get("rdrop_alpha", 1.0)
        use_msd = cfg.get("multi_sample_dropout", False)
        msd_k = cfg.get("msd_k", 4)

        # Loss
        if cfg["loss"] == "focal":
            criterion = FocalLoss(alpha=cfg.get("focal_alpha"), gamma=cfg.get("focal_gamma", 2.0)).to(device)
        elif cfg["loss"] == "soft_f1":
            criterion = MacroDoubleSoftF1Loss(num_classes=4, class_weights=cfg.get("class_weights")).to(device)
        elif cfg["loss"] == "cdw_ce":
            criterion = CDWCELoss(num_classes=4, alpha=cfg.get("cdw_alpha", 2.0),
                                  class_weights=cfg.get("class_weights")).to(device)
        elif cfg["loss"] == "weighted_ce":
            w = torch.tensor(cfg["class_weights"], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0),
                                            reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0), reduction="none")

        base_optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
        optimizer = Lookahead(base_optimizer, k=cfg.get("lookahead_k", 5),
                               alpha=cfg.get("lookahead_alpha", 0.5)) if cfg.get("lookahead") else base_optimizer
        ema = EMA(model, decay=cfg.get("ema_decay", 0.999)) if cfg.get("ema") else None
        total_steps = len(tr_ld) * cfg["epochs"] // cfg["grad_accum"]

        # Scheduler
        sched_type = cfg.get("scheduler", "cosine_warmup")
        warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
        if sched_type == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(base_optimizer, warmup_steps, total_steps)
        elif sched_type == "cosine_restarts":
            scheduler = CosineAnnealingWarmRestarts(base_optimizer, T_0=cfg.get("cosine_t0",3), T_mult=2)
        else:
            scheduler = get_linear_schedule_with_warmup(base_optimizer, warmup_steps, total_steps)
            # Override with cosine after warmup
            # (transformers' scheduler handles this as linear warmup then linear decay;
            #  for true cosine we'd need a custom scheduler, but linear warmup is close enough)

        # BF16 autocast: avoids FP16-gradient error with DeBERTa-v3; no GradScaler needed
        use_amp = device.type == "cuda"

        best_wf1, patience_counter, best_epoch = 0, 0, 0

        for epoch in range(cfg["epochs"]):
            t0 = time.time()

            # Train
            model.train()
            train_loss, steps = 0, 0
            pbar = tqdm(tr_ld, desc=f"  E{epoch+1:02d} train", leave=False,
                        unit="batch", dynamic_ncols=True)
            for batch in pbar:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                sw = batch.get("sample_weight")
                sw = sw.to(device) if sw is not None else torch.ones_like(labels, dtype=torch.float32)

                if freelb is not None:
                    # FreeLB replaces the clean pass entirely -- all K adversarial
                    # ascent steps together ARE the training signal for this batch.
                    loss = freelb.attack_and_backward(model, ids, mask, labels, sw, criterion,
                                                       use_amp, cfg["grad_accum"])
                elif use_rdrop:
                    # Two independent forward passes (different dropout masks each
                    # time since the model is in train() mode) -- replaces the
                    # single clean pass entirely, same as FreeLB structurally.
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        logits1 = model(ids, mask)
                        logits2 = model(ids, mask)
                    per_sample1 = criterion(logits1.float(), labels)
                    per_sample2 = criterion(logits2.float(), labels)
                    ce_loss = ((per_sample1 * sw).sum() / sw.sum()
                               + (per_sample2 * sw).sum() / sw.sum()) / 2
                    kl_loss = rdrop_kl(logits1.float(), logits2.float())
                    loss = (ce_loss + rdrop_alpha * kl_loss) / cfg["grad_accum"]
                    loss.backward()
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        if use_supcon or use_msd:
                            logits, pooled = model(ids, mask, return_pooled=True)
                        else:
                            logits = model(ids, mask)

                        if use_msd:
                            # Overrides `logits`/loss entirely: K dropout-masked passes
                            # through the head alone (backbone already ran once above).
                            loss, logits = multi_sample_dropout_loss(model, pooled, labels, sw,
                                                                      criterion, k=msd_k)
                        else:
                            per_sample = criterion(logits.float(), labels)
                            loss = (per_sample * sw).sum() / sw.sum()
                            if use_supcon:
                                loss = loss + supcon_alpha * supcon_loss(pooled.float(), labels)

                    loss = loss / cfg["grad_accum"]
                    loss.backward()

                    if fgm is not None:
                        fgm.attack()
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            adv_logits = model(ids, mask)
                        adv_per_sample = criterion(adv_logits.float(), labels)
                        adv_loss = (adv_per_sample * sw).sum() / sw.sum() / cfg["grad_accum"]
                        adv_loss.backward()  # accumulates on top of the clean-pass gradients
                        fgm.restore()

                    if awp is not None:
                        awp.attack()
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            adv_logits = model(ids, mask)
                        adv_per_sample = criterion(adv_logits.float(), labels)
                        adv_loss = (adv_per_sample * sw).sum() / sw.sum() / cfg["grad_accum"]
                        adv_loss.backward()  # accumulates on top of the clean-pass gradients
                        awp.restore()

                if (steps + 1) % cfg["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if ema is not None:
                        ema.update(model)

                train_loss += loss.item() * cfg["grad_accum"]
                steps += 1
                pbar.set_postfix(loss=f"{loss.item() * cfg['grad_accum']:.4f}")
            pbar.close()
            train_loss /= max(steps, 1)

            # Eval -- if EMA is active, evaluate using the averaged weights (swapped
            # in temporarily), since they're typically smoother/better-generalizing
            # than the raw noisier training weights this late in training.
            ema_backup = ema.apply_to(model) if ema is not None else None
            model.eval()
            all_preds, all_labels, val_loss = [], [], 0
            with torch.no_grad():
                for batch in va_ld:
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    logits = model(ids, mask)
                    val_loss += criterion(logits.float(), labels).sum().item()
                    all_preds.extend(logits.argmax(-1).cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            n = max(len(all_preds), 1)
            val_loss /= n
            ps = [ID2RISK[p] for p in all_preds]
            ls = [ID2RISK[l] for l in all_labels]
            wf1 = f1_score(ls, ps, labels=RISK_LEVELS, average="weighted", zero_division=0)
            mf1 = f1_score(ls, ps, labels=RISK_LEVELS, average="macro", zero_division=0)
            acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
            elapsed = time.time() - t0

            is_best = wf1 > best_wf1
            if is_best:
                best_wf1 = wf1
                best_epoch = epoch + 1
                patience_counter = 0
                # Save whatever weights were actually just evaluated to get `wf1` --
                # ema.shadow if EMA is active (matches the eval above), else the raw
                # training weights currently loaded in `model`.
                state_to_save = ema.shadow if ema is not None else model.state_dict()
                torch.save(state_to_save, os.path.join(exp_dir, f"best_fold{fold}.pt"))
            else:
                patience_counter += 1

            if ema is not None:
                ema.restore(model, ema_backup)

            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else cfg["lr"]
            print(f"    E{epoch+1:02d} ({elapsed:.0f}s) loss={train_loss:.4f}/{val_loss:.4f} "
                  f"WF1={wf1:.4f} MF1={mf1:.4f} acc={acc:.3f}"
                  f"{'  **BEST**' if is_best else ''}")

            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow({
                    "fold":fold, "epoch":epoch+1, "train_loss":f"{train_loss:.4f}",
                    "val_loss":f"{val_loss:.4f}", "val_wf1":f"{wf1:.4f}",
                    "val_mf1":f"{mf1:.4f}", "val_acc":f"{acc:.4f}",
                    "best_wf1":f"{best_wf1:.4f}", "lr":f"{lr_now:.2e}",
                    "elapsed":f"{elapsed:.0f}"})

            if patience_counter >= cfg["patience"]:
                print(f"    Early stopping at epoch {epoch+1} (patience={cfg['patience']})")
                break

        print(f"  Fold {fold+1}: best WF1={best_wf1:.4f} at epoch {best_epoch}")
        all_wf1.append(best_wf1)

        # Clean up GPU memory between folds (ema.shadow holds a full extra copy
        # of the model's weights, needs explicit cleanup same as `model` itself)
        del model, optimizer, scheduler, criterion, fgm, awp, freelb, ema
        torch.cuda.empty_cache()

    # Summary
    mean_wf1 = np.mean(all_wf1)
    std_wf1 = np.std(all_wf1)
    print(f"\n  {'='*50}")
    print(f"  {name}: WF1 = {mean_wf1:.4f} ± {std_wf1:.4f}")
    print(f"  {'='*50}\n")

    summary = {
        "experiment": name,
        "description": cfg["description"],
        "model": cfg["model_name"],
        "data": cfg["data"],
        "loss": cfg["loss"],
        "lr": cfg["lr"],
        "epochs_config": cfg["epochs"],
        "mean_wf1": float(mean_wf1),
        "std_wf1": float(std_wf1),
        "fold_wf1": [float(w) for w in all_wf1],
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Results aggregation ──────────────────────────────────────────────────────

def aggregate_results():
    """Collect all experiment summaries into one comparison file."""
    rows = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        summary_path = os.path.join(RESULTS_DIR, name, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
            if "mean_wf1" in data:  # skip foreign schemas (e.g. the generative-QA pipeline)
                rows.append(data)

    if not rows:
        print("No results found.")
        return

    rows.sort(key=lambda x: -x["mean_wf1"])
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPARISON — Subtask 1a Risk Classification")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Experiment':<40} {'WF1':>8} {'±':>6} {'Model':<30}")
    print(f"  {'-'*90}")
    for i, r in enumerate(rows):
        print(f"  {i+1:<5} {r['experiment']:<40} {r['mean_wf1']:>8.4f} "
              f"{r['std_wf1']:>6.4f} {r['model']:<30}")
    print(f"{'='*80}\n")

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # CSV version
    with open(os.path.join(RESULTS_DIR, "all_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment","description","model","data",
                                          "loss","lr","mean_wf1","std_wf1"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in w.fieldnames})

    print(f"  Results saved to {RESULTS_DIR}/all_results.json and .csv")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run risk classification experiments")
    parser.add_argument("experiments", nargs="*", help="Experiment names to run")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--dry-run", action="store_true", help="Quick pipeline test")
    parser.add_argument("--results", action="store_true", help="Show results comparison")
    args = parser.parse_args()

    if args.results:
        aggregate_results()
        return

    if args.dry_run:
        print("="*50)
        print("  DRY RUN — Testing pipeline")
        print("="*50)
        run_experiment("dry_run", DRY_RUN)
        aggregate_results()
        return

    if args.all:
        exps = list(EXPERIMENTS.keys())
    elif args.experiments:
        exps = args.experiments
    else:
        parser.print_help()
        return

    for name in exps:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            continue
        run_experiment(name, EXPERIMENTS[name])

    aggregate_results()


if __name__ == "__main__":
    main()
