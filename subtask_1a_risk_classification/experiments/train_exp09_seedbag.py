"""Seed-bag the Sub1a exp09 recipe -- same architecture/data/config, only the
random seed differs (GroupKFold split is unaffected, deterministic on
anon_user_id). Mirrors the Sub2 seed-bagging pattern (v20->v21, +0.0195 real)
which is well-motivated here too: exp09 already converges cleanly on every
fold (patience=4 fires after genuine plateaus in the existing training log --
4+ flat epochs before every fold stopped, nothing cut off on a schedule), so
"seed-bag" -- not "extend training" -- is the lever worth testing here. Also
higher-leverage than Sub2: 1a carries 0.4 composite points per unit of WF1
gain, vs Sub2's 0.3.

Reuses trainer.run_experiment() completely unmodified -- monkeypatches
trainer.SEED before calling it (Python resolves module globals at call time,
so this is safe and doesn't require editing trainer.py) and requires a
distinct --output-name so it writes to its own results/<name>/ directory,
never touching the production results/exp09_deberta_large_lr_high/
directory or its checkpoints (every submission since v9 reuses that
directory's rescored predictions unchanged).

Usage:
    python train_exp09_seedbag.py --seed 101 --output-name exp09_deberta_large_lr_high_seed101
"""

import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer
from experiment_config import EXPERIMENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp-name", default="exp09_deberta_large_lr_high",
                     help="Which EXPERIMENTS config to reuse (default: exp09, the production recipe)")
    ap.add_argument("--output-name", required=True,
                     help="Distinct results/<name>/ directory -- must differ from --exp-name so the "
                          "production checkpoints are never touched")
    args = ap.parse_args()

    if args.output_name == args.exp_name:
        raise ValueError("--output-name must differ from --exp-name -- refusing to risk writing "
                          "over the production directory")

    trainer.SEED = args.seed
    cfg = EXPERIMENTS[args.exp_name]
    print(f"Seed-bagging '{args.exp_name}' with seed={args.seed} -> results/{args.output_name}/")
    trainer.run_experiment(args.output_name, cfg)


if __name__ == "__main__":
    main()
