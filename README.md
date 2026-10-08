# ALEXIS — ShiftGuard: IEEE BigData 2026 Cup, Explainable Suicide Risk Detection

Source code accompanying our team report (`../report/paper.pdf`, or compile
`../report/paper.tex`), *"ShiftGuard: A Trust-Aware System for Explainable
Suicide Risk Detection under Population Shift."* This
package covers the full ALEXIS system across all three subtasks (risk
classification, evidence extraction, factor identification) and the
ShiftGuard shift-diagnosis and gating framework.

A companion paper, *"Target-Aware Validation under Population Shift:
Diagnosing Selection Bias in Small-Cohort Suicide Risk Detection"* (MHSM
2026 @ IEEE ICDM 2026), develops the shift-diagnosis and importance-weighted
gating methodology in depth for Subtasks 1a/1b only. This repository
reproduces that methodology (see `subtask_1a_risk_classification/experiments/`)
and additionally covers Subtask 2 and the full retained system.

## What is (and isn't) in this package

This package contains **source code only**. It deliberately does **not**
include:

- the competition's own training/leaderboard data (`train.xlsx`,
  `leaderboard.xlsx`) — this is the organizers' data, not ours to
  redistribute;
- the external RSD-15K corpus used for several experiments in the report —
  a separately-licensed dataset obtained directly from its authors, also not
  ours to redistribute;
- any trained model checkpoints, cached embeddings, or prediction outputs —
  these are large, fully regenerable by running the code below, and several
  of them are derived from the two datasets above.

See **Data** below for how to obtain both datasets yourself; both are
freely available to researchers through the normal channels, just not
bundled here.

## Repository structure

```
code/
  subtask_1a_risk_classification/
    config.py, dataset.py, model.py, train.py, predict.py   -- baseline pipeline
    experiments/
      trainer.py, experiment_config.py                       -- main training harness
      (all Sub1a experiment, diagnostic, and analysis scripts described in the report)
  subtask_1b_evidence_extraction/
    config.py, dataset.py, model.py, crf.py, train.py, predict.py
    predict_ensemble_1b.py   -- N-way checkpoint ensemble (used for the retained v37 composition)
    (composition search/build scripts, diagnostic and analysis scripts)
  subtask_2_factor_identification/
    config.py, dataset_nli_finetune.py, model_nli_finetune.py
    train_nli_finetune_kfold.py, predict_nli_finetune_kfold.py   -- production pipeline (LP-FT NLI cross-encoder)
    (all other tried architectures -- GCN, dual-encoder, contrastive NLI, chain models,
     seed-bagging, alternate NLI checkpoints -- included for transparency; none of these
     were used in the retained submission, see the report's Section on Subtask 2)
  requirements.txt
  README.md   -- this file
report/
  paper.tex   -- IEEE conference report source (see report/README.md to compile)
```

## Setup (Linux, Windows, or macOS)

Requires Python 3.10+ and, for realistic training times, a CUDA-capable GPU
(the production models are DeBERTa-v3-large scale; CPU-only will run but is
very slow). All code is plain Python using `os.path`/`pathlib`-style joins
throughout, so it is not Linux-specific.

```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

If your PyTorch build needs a specific CUDA version, install it first from
https://pytorch.org/get-started/locally/ before `pip install -r requirements.txt`
(the pinned `torch>=2.0` will otherwise pull a default build).

## Data

All three subtask folders expect a `data/` subdirectory:

```
subtask_1a_risk_classification/data/train.xlsx
subtask_1a_risk_classification/data/leaderboard.xlsx
subtask_1b_evidence_extraction/data/train.xlsx
subtask_1b_evidence_extraction/data/leaderboard.xlsx
subtask_2_factor_identification/data/train.xlsx
subtask_2_factor_identification/data/leaderboard.xlsx
```

Obtain these from the competition organizers (they are provided to
registered participants of the Explainable Suicide Risk Detection
challenge) and place them as shown above.

For the RSD-15K-based experiments (`train_rsd15k_two_stage.py`,
`train_rsd15k_joint.py`, `rsd15k_controlled_shift.py`,
`eval_exp09_on_rsd15k.py`, `calibrate_on_rsd15k.py`, and the
`rsd15k_check_*.py` scripts), request access to RSD-15K from its authors
(Zheng, Tao, and Zhou, *"RSD-15K: A Large-Scale User-Level Annotated
Dataset for Suicide Risk Detection on Social Media,"* IEEE ICDEW 2025) via
their stated application process (non-commercial research use only, no
redistribution) and place the resulting CSV at `data/RSD-15K/RSD-15K.csv` (relative to the
`code/` directory, i.e. as a sibling of both subtask folders), with columns
`users, text, sentiment, time`. Every script that uses it reads from this
one path.

flan-t5-large and flan-t5-xl are downloaded automatically via the
HuggingFace hub (`google/flan-t5-large`, `google/flan-t5-xl`) the first
time a script that needs them is run — no manual download required given
normal internet access.

## Reproducing the main results

All commands below assume you are in the `code/` directory with the venv
activated and data placed as above.

**Subtask 1a production model (`exp09`, the classifier underlying every
submission's Sub1a component unless otherwise noted):**
```bash
cd subtask_1a_risk_classification/experiments
python trainer.py exp09_deberta_large_lr_high
```
Other configurations in `experiment_config.py` (CORAL, FGM, self-training,
FreeLB, R-Drop, the domain-adaptive-pretraining variant, etc.) are run the
same way, e.g. `python trainer.py exp38_deberta_coral_weighted`.

**Subtask 1b production model.** The retained system (v37) is a 2-way
ensemble of the *denoised* family (seed 42) and the *CRF* family trained
with seed 100 (not the larger family search below -- this is the exact
composition behind every reported real/final result):
```bash
cd subtask_1b_evidence_extraction
python train.py --denoise-indicators --output-dir outputs_denoised          # 'denoised' family, seed 42
python train.py --use-crf --seed 100 --output-dir outputs_crf_seed100       # 'crf' family, seed 100
python predict_ensemble_1b.py --dirs outputs_denoised outputs_crf_seed100 \
    --risk-csv ../subtask_1a_risk_classification/submissions/<exp09_seqrescored>.csv \
    --out evidence_predictions_v37.csv
```
The broader 4-family composition search (`vanilla`, `crf`, `denoised`,
`crf_denoised`, searched over 25 subsets) was tried during development and
is included below for completeness, but it is a *different*, not-adopted
candidate (see the report's differential-submission analysis) -- do not use
it to reproduce the retained system:
```bash
python train.py --output-dir outputs                                    # 'vanilla' family
python train.py --use-crf --output-dir outputs_crf                      # 'crf' family
python train.py --use-crf --denoise-indicators --output-dir outputs_crf_denoised  # 'crf_denoised' family
python search_ensemble_composition.py    # searches compositions of the trained families
python build_composition_submission.py   # builds the best-found composition's leaderboard predictions
```

**Subtask 2 production model (NLI cross-encoder, LP-FT, single seed):**
```bash
cd subtask_2_factor_identification
python train_nli_finetune_kfold.py --output-dir outputs_nli_finetune_kfold
python predict_nli_finetune_kfold.py --output-dir outputs_nli_finetune_kfold
```
This is the exact pipeline behind the retained system's real Subtask 2
score; a seed-bagging variant (`train_nli_seedbag.py`, `predict_nli_seedbag.py`)
and several alternative formulations (GCN, dual-encoder, contrastive NLI,
chain models, alternate NLI checkpoints) were tried during development and
are included for transparency, but none were adopted into any real
submission -- see the report for details.

**Population-shift diagnostics (Section V of the report):**
```bash
cd subtask_1a_risk_classification/experiments
python population_shift_diagnostic.py          # adversarial validation, train vs. leaderboard
python importance_weighted_validation.py       # the target-aware validation gate
python eval_exp09_on_rsd15k.py                 # external confirmation via RSD-15K (needs RSD-15K, see Data)
python rsd15k_controlled_shift.py              # the controlled shift benchmark (needs RSD-15K)

cd ../../subtask_1b_evidence_extraction
python population_shift_diagnostic_1b.py       # Sub1b-native adversarial validation
python importance_weighted_validation_1b_v2.py # Sub1b-native validation gate
```

**Negative results (Section VI of the report):**
```bash
cd subtask_1a_risk_classification/experiments
python train_dann_domain_adversarial.py   # domain-adversarial training
python confidence_routed_features.py      # confidence-routed engineered features
python train_rsd15k_two_stage.py          # RSD-15K two-stage transfer learning (needs RSD-15K)
python train_rsd15k_joint.py              # RSD-15K joint training (needs RSD-15K)
python calibrate_on_rsd15k.py             # RSD-15K post-hoc calibration correction (needs RSD-15K)
```

## Notes

- GPU memory: the DeBERTa-v3-large scripts were developed against a single
  GPU with roughly 20GB+ free VRAM at batch size 16 (`microsoft/deberta-v3-large`
  fits comfortably); reduce `BATCH_SIZE` in the relevant config if you hit
  out-of-memory errors on a smaller GPU.
- All cross-validation uses 5-fold `GroupKFold` on `anon_user_id`, so no
  user's posts appear in both a fold's training and validation split.
- Filenames beginning `exp*_qa_sft*` explore Subtask 1a as text-to-text QA
  with flan-t5 rather than classification; these, along with the CORAL,
  NLI-ordinal, and BIOES-tagging variants, are reported in the paper as
  either negative results or intermediate steps, included here for
  transparency and reproducibility rather than as the recommended pipeline.
