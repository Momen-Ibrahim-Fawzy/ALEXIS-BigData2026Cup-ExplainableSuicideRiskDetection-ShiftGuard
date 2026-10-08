"""Extends importance_weighted_validation.py's retrospective test to v42 and
v43 -- two more real, known outcomes (-0.0107 and -0.0079 vs v37), using
their exact original ensemble member compositions. Gives 4 total 1a data
points (exp42, flan-t5 ensemble, v42, v43) instead of 2, to properly assess
whether the importance-weighting methodology is reliable rather than a
one-off coincidence.

v42 = exp09 + FGM(32) + selftrain(33) + FreeLB(36) + CORAL(38) + RDrop(39), uniform average
v43 = exp09 + selftrain(33) + CORAL(38), uniform average

Usage:
    python importance_weighted_validation_v42_v43.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from trainer import Classifier, TextDataset, RISK2ID, DATA_DIR
from coral_model import CoralDualHeadClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # subtask_1a_risk_classification
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SEQ_DIR = os.path.join(RESULTS_DIR, "exp09_deberta_large_lr_high", "sequence_data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-large"

MEMBERS = {
    "exp09":     ("exp09_deberta_large_lr_high", "plain"),
    "exp32_fgm": ("exp32_deberta_large_fgm", "plain"),
    "exp33_selftrain": ("exp33_deberta_large_selftrain", "plain"),
    "exp36_freelb": ("exp36_deberta_large_freelb", "plain"),
    "exp38_coral": ("exp38_deberta_coral_weighted", "coral"),
    "exp39_rdrop": ("exp39_deberta_large_rdrop", "plain"),
}

V42_MEMBERS = ["exp09", "exp32_fgm", "exp33_selftrain", "exp36_freelb", "exp38_coral", "exp39_rdrop"]
V43_MEMBERS = ["exp09", "exp33_selftrain", "exp38_coral"]


def fit_adversarial_weights():
    train_emb = np.load(os.path.join(SEQ_DIR, "train_oof_embeddings.npy"))
    lb_emb = np.load(os.path.join(SEQ_DIR, "leaderboard_embeddings.npy"))
    X = np.concatenate([train_emb, lb_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb)), np.ones(len(lb_emb))])
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit((X - mu) / sigma, y)
    p_lb = np.clip(clf.predict_proba((train_emb - mu) / sigma)[:, 1], 0.02, 0.98)
    w = p_lb / (1 - p_lb)
    w = w / w.mean()
    return np.clip(w, 0.1, 10.0)


@torch.no_grad()
def get_member_oof_probs(member_key, df, groups, gkf, tokenizer):
    exp_dir_name, arch = MEMBERS[member_key]
    cache_path = os.path.join(RESULTS_DIR, f"{member_key}_train_oof_probs.npy")
    if os.path.exists(cache_path):
        print(f"    (using cached {member_key} true-OOF probs)")
        return np.load(cache_path)

    exp_dir = os.path.join(RESULTS_DIR, exp_dir_name)
    all_probs = np.zeros((len(df), 4))
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        va_df = df.iloc[va_i]
        ckpt_path = os.path.join(exp_dir, f"best_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

        if arch == "coral":
            model = CoralDualHeadClassifier(MODEL_NAME).to(DEVICE).float()
        else:
            model = Classifier(MODEL_NAME).to(DEVICE).float()
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        ds = TextDataset(va_df["post"].tolist(), [RISK2ID[r] for r in va_df["risk_norm"]], tokenizer, 512)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
        probs = []
        for batch in ld:
            ids = batch["input_ids"].to(DEVICE); mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if arch == "coral":
                    sl, cl = model(ids, mask)
                    p = model.combined_probs(sl.float(), cl.float())
                else:
                    logits = model(ids, mask)
                    p = torch.softmax(logits.float(), dim=-1)
            probs.append(p.cpu().numpy())
        all_probs[va_i] = np.concatenate(probs, axis=0)
        del model
        torch.cuda.empty_cache()
        print(f"      {member_key} fold {fold} done")

    np.save(cache_path, all_probs)
    return all_probs


def main():
    print(f"{'='*70}\n  Fitting adversarial-validation importance weights\n{'='*70}")
    weights = fit_adversarial_weights()
    print(f"  weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_excel(os.path.join(DATA_DIR, "train.xlsx"), engine="openpyxl")
    df["risk_norm"] = df["suicide risk"].str.strip().str.lower()
    df = df.reset_index(drop=True)
    true_labels = np.array([RISK2ID[r] for r in df["risk_norm"]])
    groups = df["anon_user_id"].values
    gkf = GroupKFold(n_splits=5)

    print(f"\n{'='*70}\n  Extracting true-OOF probs for every needed member\n{'='*70}")
    member_probs = {}
    for key in sorted(set(V42_MEMBERS) | set(V43_MEMBERS)):
        print(f"  {key}:")
        member_probs[key] = get_member_oof_probs(key, df, groups, gkf, tokenizer)

    exp09_probs = member_probs["exp09"]
    v42_probs = np.mean([member_probs[k] for k in V42_MEMBERS], axis=0)
    v43_probs = np.mean([member_probs[k] for k in V43_MEMBERS], axis=0)

    candidates = {
        "exp09 (baseline)": exp09_probs,
        "v42 6-way (REAL: -0.0107)": v42_probs,
        "v43 3-way (REAL: -0.0079)": v43_probs,
    }

    print(f"\n{'='*70}\n  UNWEIGHTED vs IMPORTANCE-WEIGHTED true-OOF WF1\n{'='*70}")
    results = {}
    for name, probs in candidates.items():
        preds = probs.argmax(axis=1)
        u = f1_score(true_labels, preds, average="weighted", zero_division=0)
        w = f1_score(true_labels, preds, average="weighted", sample_weight=weights, zero_division=0)
        results[name] = (u, w)
        print(f"  {name}\n    unweighted: {u:.4f}   weighted: {w:.4f}")

    eu, ew = results["exp09 (baseline)"]
    print(f"\n{'='*70}\n  DELTAS vs exp09\n{'='*70}")
    for name, (u, w) in results.items():
        if name == "exp09 (baseline)":
            continue
        du, dw = u - eu, w - ew
        print(f"  {name}\n    unweighted delta: {du:+.4f}\n    weighted delta:   {dw:+.4f}")
        if du > 0 and dw < 0:
            print("    -> WEIGHTING FLIPPED THE SIGN: would have correctly predicted the real regression!")
        elif du > 0 and dw > 0 and dw < du:
            print("    -> weighting reduced the apparent gain (didn't flip, but less misleading)")
        elif du > 0 and dw >= du:
            print("    -> weighting did NOT help")
        else:
            print("    -> (unweighted was already negative or flat)")


if __name__ == "__main__":
    main()
