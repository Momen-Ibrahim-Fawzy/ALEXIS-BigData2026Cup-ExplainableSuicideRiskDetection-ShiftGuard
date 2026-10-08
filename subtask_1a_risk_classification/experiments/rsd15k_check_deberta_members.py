"""RSD-15K retrospective check, part 1: the 5 DeBERTa members needed for v42
(6-way: exp09+FGM+selftrain+FreeLB+CORAL+RDrop) and v43 (3-way: exp09+
selftrain+CORAL), plus exp42 (long-context standalone). exp09 itself is
already cached (exp09_rsd15k_probs.npy from eval_exp09_on_rsd15k.py).

Tests whether "does this candidate's RSD-15K WF1 beat exp09's" would have
correctly flagged the real regressions -- a labeled, external, much larger
(14,613 vs 378) alternative to the adversarial-classifier-based importance
weighting, which only hit 2/5 as a binary gate.

Usage:
    python rsd15k_check_deberta_members.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, RESULTS_DIR
from coral_model import CoralDualHeadClassifier

RSD_PATH = "../../data/RSD-15K/RSD-15K.csv"
MODEL_NAME = "microsoft/deberta-v3-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
BATCH_SIZE = 16

MEMBERS = {
    "exp32_fgm": ("exp32_deberta_large_fgm", "plain", 512),
    "exp33_selftrain": ("exp33_deberta_large_selftrain", "plain", 512),
    "exp36_freelb": ("exp36_deberta_large_freelb", "plain", 512),
    "exp38_coral": ("exp38_deberta_coral_weighted", "coral", 512),
    "exp39_rdrop": ("exp39_deberta_large_rdrop", "plain", 512),
    "exp42_longctx": ("exp42_deberta_large_long_context_clean", "plain", 768),
}


@torch.no_grad()
def predict_5fold_avg(exp_dir, arch, max_length, ds):
    ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    fold_probs = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(RESULTS_DIR, exp_dir, f"best_fold{fold}.pt")
        if arch == "coral":
            model = CoralDualHeadClassifier(MODEL_NAME).to(DEVICE).float()
        else:
            model = Classifier(MODEL_NAME).to(DEVICE).float()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        probs_this_fold = []
        for batch in ld:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if arch == "coral":
                    sl, cl = model(ids, mask)
                    p = model.combined_probs(sl.float(), cl.float())
                else:
                    logits = model(ids, mask)
                    p = torch.softmax(logits.float(), dim=-1)
            probs_this_fold.append(p.cpu().numpy())
        fold_probs.append(np.concatenate(probs_this_fold, axis=0))
        print(f"    fold {fold} done", flush=True)
        del model
        torch.cuda.empty_cache()

    return np.mean(fold_probs, axis=0)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    rsd = pd.read_csv(RSD_PATH)
    rsd["risk_norm"] = rsd["sentiment"].str.strip().str.lower()
    true_labels = np.array([RISK2ID[r] for r in rsd["risk_norm"]])

    for key, (exp_dir, arch, max_len) in MEMBERS.items():
        cache_path = os.path.join(RESULTS_DIR, f"{key}_rsd15k_probs.npy")
        if os.path.exists(cache_path):
            print(f"{key}: cached, skipping")
            continue
        print(f"\n{'='*70}\n  {key} ({exp_dir}, arch={arch}, max_len={max_len})\n{'='*70}", flush=True)
        ds = TextDataset(rsd["text"].astype(str).tolist(), true_labels.tolist(), tokenizer, max_len)
        probs = predict_5fold_avg(exp_dir, arch, max_len, ds)
        np.save(cache_path, probs)
        wf1 = f1_score(true_labels, probs.argmax(1), average="weighted", zero_division=0)
        print(f"  {key} RSD-15K WF1: {wf1:.4f}")


if __name__ == "__main__":
    main()
