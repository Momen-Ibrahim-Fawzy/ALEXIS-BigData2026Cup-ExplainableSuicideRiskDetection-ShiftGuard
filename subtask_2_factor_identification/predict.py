"""Subtask 2: Generate leaderboard factor predictions with tuned thresholds."""

import os, json, argparse, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from config import *
from dataset import load_leaderboard, get_tokenizer, FactorDataset
from model import FactorClassifier


def predict(output_dir=None):
    if output_dir:
        globals()["OUTPUT_DIR"] = os.path.abspath(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    lb_df = load_leaderboard()
    ds = FactorDataset(lb_df, tokenizer, is_test=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

    # Load thresholds
    thresh_path = os.path.join(OUTPUT_DIR, "factor_thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            td = json.load(f)
        thresholds = np.array([td["thresholds"].get(f, 0.5) for f in FACTOR_LIST])
        print(f"Using tuned thresholds (Macro F1={td['tuned_macro_f1']:.4f})")
    else:
        thresholds = np.full(NUM_FACTORS, 0.5)
        print("Using default thresholds (0.5)")

    # Collect models
    model_paths = [os.path.join(OUTPUT_DIR, f"fold_{f}", "best.pt")
                   for f in range(N_FOLDS)
                   if os.path.exists(os.path.join(OUTPUT_DIR, f"fold_{f}", "best.pt"))]
    print(f"Ensembling {len(model_paths)} models")

    all_probs = []
    for p in model_paths:
        m = FactorClassifier().to(device).float()
        m.load_state_dict(torch.load(p, weights_only=False, map_location=device)["model"])
        m.eval()
        fold_p = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ttype = batch.get("token_type_ids")
                if ttype is not None: ttype = ttype.to(device)
                logits = m(ids, mask, ttype)
                fold_p.append(torch.sigmoid(logits).cpu().numpy())
        all_probs.append(np.concatenate(fold_p))

    avg = np.mean(all_probs, axis=0)
    # Save raw probabilities for downstream LLM ensemble (v3)
    np.save(os.path.join(OUTPUT_DIR, "leaderboard_probs.npy"), avg)
    preds = (avg >= thresholds).astype(int)

    # Build output
    rows = []
    for i in range(len(lb_df)):
        factors = [FACTOR_LIST[j] for j in range(NUM_FACTORS) if preds[i,j]]
        rows.append({"row_id": lb_df.iloc[i]["row_id"],
                      "factors": str(factors) if factors else "[]"})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "factor_predictions.csv"), index=False)

    out["n"] = out["factors"].apply(lambda x: len(eval(x)) if x != "[]" else 0)
    print(f"Saved. Mean factors/post: {out['n'].mean():.1f}, "
          f"zero: {(out['n']==0).sum()}/{len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: outputs/)")
    args = parser.parse_args()
    predict(output_dir=args.output_dir)
