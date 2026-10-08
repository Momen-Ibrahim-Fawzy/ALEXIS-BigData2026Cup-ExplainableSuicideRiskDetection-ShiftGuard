"""
Experiment configurations for Subtask 1a: Risk Classification.

Each experiment is a dict defining model, data, training, and loss settings.
The trainer reads this config and runs the experiment end-to-end.
"""

import os

EXPERIMENTS = {

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 1: MODEL BACKBONE
    # ═══════════════════════════════════════════════════════════════════════════

    "exp01_deberta_large": {
        "description": "DeBERTa-v3-large baseline with weighted CE",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp02_deberta_base": {
        "description": "DeBERTa-v3-base (smaller, faster, compare capacity)",
        "model_name": "microsoft/deberta-v3-base",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp03_roberta_large": {
        "description": "RoBERTa-large (strong general baseline)",
        "model_name": "roberta-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp04_modernbert_base": {
        "description": "ModernBERT-base (2024 efficient encoder, replaces mental-roberta not cached)",
        "model_name": "answerdotai/ModernBERT-base",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    # COMMENTED OUT — mental-roberta-base requires internet / HuggingFace access.
    # Run on a server with internet: bash run_one.sh exp04_mental_roberta
    # "exp04_mental_roberta": {
    #     "description": "MentalRoBERTa (domain-adapted on Reddit mental health)",
    #     "model_name": "mental/mental-roberta-base",
    #     "data": "original",
    #     "loss": "weighted_ce",
    #     "class_weights": [0.669, 0.788, 1.045, 3.586],
    #     "label_smoothing": 0.05,
    #     "lr": 2e-5,
    #     "scheduler": "cosine_warmup",
    #     "warmup_ratio": 0.1,
    #     "epochs": 32,
    #     "batch_size": 32,
    #     "grad_accum": 1,
    #     "patience": 4,
    #     "max_length": 512,
    # },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 2: DATA AUGMENTATION
    # ═══════════════════════════════════════════════════════════════════════════

    "exp05_deberta_large_oversample_gpt": {
        "description": "DeBERTa-v3-large + GPT-OSS oversampled data (all 809)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_all",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp06_deberta_large_oversample_verified": {
        "description": "DeBERTa-v3-large + only both-model-verified oversamples",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_verified",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp05b_deberta_large_oversample_claude": {
        "description": "DeBERTa-v3-large + Claude oversampled data (809)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_claude",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp05c_deberta_large_oversample_llama": {
        "description": "DeBERTa-v3-large + LLAMA oversampled data (809)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_llama",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp05d_deberta_large_oversample_all_3_sources": {
        "description": "DeBERTa-v3-large + ALL 3 oversample sources (2427 synthetic)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_all_sources",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 3: LOSS FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    "exp07_deberta_large_focal": {
        "description": "DeBERTa-v3-large + focal loss (gamma=2, alpha=class weights)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "focal",
        "focal_gamma": 2.0,
        "focal_alpha": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.0,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp08_deberta_large_no_weights": {
        "description": "DeBERTa-v3-large + plain CE (no class weights, test if weights help)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "ce",
        "label_smoothing": 0.1,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 4: LEARNING RATE & SCHEDULER
    # ═══════════════════════════════════════════════════════════════════════════

    "exp09_deberta_large_lr_high": {
        "description": "DeBERTa-v3-large + higher LR (2e-5) + linear scheduler",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp46_domain_adaptive_mlm_v2": {
        "description": "exp09's exact recipe, starting from the v2 domain-adapted backbone "
                        "(bottom 20/24 layers frozen, held-out MLM validation + early stopping, "
                        "collapse check passed post-training, unlike the v1 attempt which collapsed)",
        "model_name": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "domain_adapted_deberta_mlm_v2", "final"),
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp10_deberta_large_cosine_restarts": {
        "description": "DeBERTa-v3-large + cosine with warm restarts (T0=3)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_restarts",
        "cosine_t0": 3,
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 5,
        "max_length": 512,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 5: BEST COMBINATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    "exp11_deberta_large_focal_oversample": {
        "description": "DeBERTa-v3-large + focal loss + all 3 oversample sources (best combo)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_all_sources",
        "loss": "focal",
        "focal_gamma": 2.0,
        "focal_alpha": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.0,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 20,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 5,
        "max_length": 512,
    },

    "exp12_deberta_large_long_context": {
        "description": "DeBERTa-v3-large + max_length=768 (capture longer attempt posts)",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 8,
        "grad_accum": 2,
        "patience": 4,
        "max_length": 768,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 6: MENTAL HEALTH PRE-TRAINED / FINE-TUNED BACKBONES
    # ═══════════════════════════════════════════════════════════════════════════

    "exp13_mental_roberta_base": {
        "description": "MentalRoBERTa-base (pre-trained on mental health Reddit posts)",
        "model_name": "mental/mental-roberta-base",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp14_mental_roberta_large": {
        "description": "MentalRoBERTa-large (large mental health pre-trained model)",
        "model_name": "AIMH/mental-roberta-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 1e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp15_roberta_mental_health": {
        "description": "RoBERTa fine-tuned on mental health (lishaangral general mental health)",
        "model_name": "lishaangral/roberta-mental-health-v2",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp16_roberta_depression": {
        "description": "MentalRoBERTa fine-tuned on depression (welsachy) — transfer to risk",
        "model_name": "welsachy/mental-roberta-base-finetuned-depression",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp17_mental_roberta_stress": {
        "description": "MentalRoBERTa fine-tuned on stress (dstefa) — transfer to risk",
        "model_name": "dstefa/mental-roberta_stress_classification",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp18_emotion_distilroberta": {
        "description": "Emotion DistilRoBERTa (Hartmann, 762k dls) — emotion → risk transfer",
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp19_go_emotions_roberta": {
        "description": "RoBERTa GoEmotions (SamLowe, 1.2M dls) — emotion → risk transfer",
        "model_name": "SamLowe/roberta-base-go_emotions",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp20_suicidal_electra": {
        "description": "Suicidal ELECTRA (gooohjy) — trained directly on suicidal content",
        "model_name": "gooohjy/suicidal-electra",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp21_mental_longformer": {
        "description": "Mental Longformer base (AIMH, up to 4096 tokens) — full post context",
        "model_name": "AIMH/mental-longformer-base-4096",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 8,
        "grad_accum": 4,
        "patience": 4,
        "max_length": 1024,
    },

    "exp22_mental_xlnet": {
        "description": "Mental XLNet base (AIMH) — autoregressive attention on mental health text",
        "model_name": "AIMH/mental-xlnet-base-cased",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp23_bert_suicidal_ideation": {
        "description": "BERT fine-tuned for suicidal ideation (HubertBaginski) — direct task transfer",
        "model_name": "HubertBaginski/bert-suicidal-ideation",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp24_mental_bert_base": {
        "description": "Mental-BERT base uncased — BERT pre-trained on mental health corpora (mental/mental-bert-base-uncased)",
        "model_name": "mental/mental-bert-base-uncased",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "cosine_warmup",
        "warmup_ratio": 0.1,
        "epochs": 32,
        "batch_size": 32,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 7: TARGETED CONTRASTIVE AUGMENTATION (2026-07-25 confusion-matrix diagnostic)
    # ═══════════════════════════════════════════════════════════════════════════

    "exp26_deberta_large_contrastive_augment": {
        "description": "exp09's exact recipe + 60 hand-authored contrastive minimal pairs "
                        "targeting two diagnosed failure patterns: (A) interpersonal violence "
                        "done TO the poster confused with self-directed harm, and (B) non-current "
                        "framing (dream/past/hypothetical/negated) confused with a current real "
                        "action. Not generic class-volume oversampling (already tried repeatedly, "
                        "never helped) -- a small, surgical set isolating the exact distinction "
                        "the confusion matrix showed exp09 was missing.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_contrastive",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp27_deberta_large_lrhigh_longcontext": {
        "description": "exp09's LR/scheduler (2e-5, linear warmup) + exp12's long context "
                        "(max_length=768) -- never tested together. exp12 (+0.0075 over the "
                        "1e-5/cosine baseline) predates exp09's LR finding; exp09 (+0.0005 over "
                        "the same baseline) never used the longer context.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 8,
        "grad_accum": 2,
        "patience": 4,
        "max_length": 768,
    },

    "exp28_deberta_large_lrhigh_patience8": {
        "description": "exp09's exact recipe, patience 4 -> 8. EXPERIMENT_REPORT.md flagged "
                        "that patience=4 may cut folds off before convergence but this was never "
                        "tested -- isolates the patience question alone (same LR/context as exp09, "
                        "unlike exp27 which changes context length too).",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 8,
        "max_length": 512,
    },

    "exp29_deberta_large_longcontext_patience8": {
        "description": "Stacks the two independent 2026-07-25 wins: exp27's long context "
                        "(max_length=768, true-OOF+rescore 0.8113, +0.0035 vs production) and "
                        "exp28's extended patience (patience=8, true-OOF+rescore 0.8145, +0.0067 "
                        "vs production). Neither change overlaps mechanistically (context window "
                        "vs. early-stopping budget), so testing whether they stack additively -- "
                        "same discipline as sequence-rescoring + Sub2's ensemble stacking earlier "
                        "in this project.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 8,
        "grad_accum": 2,
        "patience": 8,
        "max_length": 768,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 8: PUSHING THE ONE CONFIRMED WIN FURTHER (2026-07-25, post exp28)
    # ═══════════════════════════════════════════════════════════════════════════

    "exp30_deberta_large_lrhigh_patience12": {
        "description": "exp28's exact recipe, patience 8 -> 12. exp28 (patience=4->8) was the "
                        "first confirmed real Sub1a gain in the whole project's history (true "
                        "OOF+rescore 0.8145, +0.0067 over production); this checks whether there's "
                        "more room past patience=8 or whether it was already the sweet spot -- a "
                        "cheap, single-variable extension of a technique we know works, not a new "
                        "architecture (which has failed 8 independent times across this project).",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 40,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 12,
        "max_length": 512,
    },

    "exp31_deberta_large_lr3e5_patience8": {
        "description": "exp28's recipe (patience=8) + lr=3e-5 instead of 2e-5. The original axis-4 "
                        "sweep only ever tested {1e-5, 2e-5} and found 2e-5 better; 3e-5 was never "
                        "tried, and never combined with the patience finding. Isolates the LR "
                        "question alone at the new best patience setting.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 3e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 8,
        "max_length": 512,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # AXIS 9: ADVERSARIAL TRAINING (2026-08-06) — never tried in this project.
    # Transferred from a sibling project (DialectSentEval Arabic dialect sentiment
    # classification) where FGM was the single best individual technique in an
    # analogous small-imbalanced-text 4-class-ish setup on the same backbone
    # family, and is independently literature-backed specifically for suicide-risk
    # assessment ("Robust suicide risk assessment on social media via deep
    # adversarial learning"). Isolates FGM alone against the one Sub1a recipe with
    # a genuine real-leaderboard confirmation (exp09, patience=4) -- deliberately
    # NOT stacked on exp28 (patience=8), since exp28 itself has only ever been
    # validated OOF, never confirmed for real (unlike exp09, which underlies the
    # confirmed-0.7805 production submission).
    # ═══════════════════════════════════════════════════════════════════════════

    "exp32_deberta_large_fgm": {
        "description": "exp09's exact recipe (lr=2e-5, patience=4, weighted CE + label "
                        "smoothing 0.05) + FGM adversarial training: after the clean forward/ "
                        "backward pass, perturb the backbone's word-embedding weights along the "
                        "normalized loss-gradient direction (epsilon=1.0), take a second forward/ "
                        "backward pass on the perturbed embeddings (gradients accumulate on top "
                        "of the clean ones), then restore the embeddings before the optimizer "
                        "step. Encourages robustness to small embedding-space perturbations -- a "
                        "proxy for the lexical/framing variation between otherwise-similar posts "
                        "at adjacent risk levels (the exact boundary confusion exp26's contrastive "
                        "pairs targeted directly; this targets it structurally instead).",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "fgm": True,
        "fgm_epsilon": 1.0,
    },

    "exp33_deberta_large_selftrain": {
        "description": "exp09's exact recipe + self-training: the 244/378 leaderboard posts "
                        "exp09's own 5-fold ensemble is most confident about (avg softmax "
                        "confidence >= 0.75) are added to training as pseudo-labeled rows at "
                        "0.7x loss weight (see pseudo_label_leaderboard.py), train-only via the "
                        "existing anon_user_id='SYN' protection -- never enters a validation "
                        "fold. Never tried in this project. Directly transfers the single most "
                        "reliable compounding technique from a sibling project's Subtask 1 "
                        "(DialectSentEval), where 5 successive self-training rounds each "
                        "improved real leaderboard F1 with no reversal. Isolated against plain "
                        "exp09 (not stacked on FGM yet) to measure this one variable alone.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "pseudo_leaderboard",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp34_deberta_large_supcon": {
        "description": "exp09's exact recipe + Supervised Contrastive Loss (Khosla et al. "
                        "2020) added on top of the CE objective: total = CE + 0.3*SupCon. "
                        "Pulls same-class pooled embeddings together, pushes different-class "
                        "ones apart within each batch. Structurally distinct from FGM/AWP "
                        "(representation-space regularization, not adversarial robustness) -- "
                        "the least likely of today's new techniques to fail for the same "
                        "reason FGM did, since it's not in the same mechanism family.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "supcon": True,
        "supcon_alpha": 0.3,
    },

    "exp35_deberta_large_awp": {
        "description": "exp09's exact recipe + Adversarial Weight Perturbation (Wu et al. "
                        "2020): perturbs the backbone's own linear/attention WEIGHTS (not "
                        "input embeddings, unlike FGM) along the normalized loss-gradient "
                        "direction, scaled relative to each weight tensor's own norm and "
                        "clipped to a relative epsilon ball. Excludes LayerNorm and embedding "
                        "weights (standard AWP scope). Mechanically close to FGM (adversarial "
                        "robustness family) so a similar (negative) outcome is plausible, but "
                        "weight-space vs. embedding-space perturbation is a real difference "
                        "worth isolating rather than assuming.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "awp": True,
        "awp_lr": 1e-3,
        "awp_eps": 1e-3,
    },

    "exp36_deberta_large_freelb": {
        "description": "exp09's exact recipe + FreeLB (Zhu et al. 2020): multi-step (K=2) "
                        "embedding-space adversarial ascent, accumulating gradient across all "
                        "K adversarial views of each batch (vs. FGM's single step), simulating "
                        "a larger effective batch against a stronger adversary. Replaces the "
                        "clean forward/backward entirely (no separate clean pass, unlike "
                        "FGM/AWP) -- all K steps together are the training signal.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "freelb": True,
        "freelb_k": 2,
        "freelb_lr": 1e-2,
        "freelb_eps": 1e-1,
    },

    "exp37_deberta_large_lookahead_ema": {
        "description": "exp09's exact recipe + Lookahead optimizer wrapper (Zhang et al. "
                        "2019, k=5 fast steps per slow step, alpha=0.5) + EMA weight "
                        "averaging (decay=0.999, evaluated/checkpointed using the averaged "
                        "weights, not the raw noisier training weights). Optimization-"
                        "stability-focused, not robustness-focused -- a different failure "
                        "mode than FGM/AWP/self-training, all of which target generalization "
                        "via perturbation or more data rather than smoother optimization.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "lookahead": True,
        "lookahead_k": 5,
        "lookahead_alpha": 0.5,
        "ema": True,
        "ema_decay": 0.99,  # 0.999 (half-life ~693 steps) was miscalibrated for this
                            # dataset's short training horizon (~800-1600 total steps
                            # before patience triggers) -- caused fold 1 of the first
                            # attempt to lock in a still-near-random checkpoint as "best"
                            # before the EMA had meaningfully diverged from init (WF1
                            # collapsed to 0.0238 vs the other 4 folds' 0.72-0.81).
    },

    "exp39_deberta_large_rdrop": {
        "description": "exp09's exact recipe + R-Drop (Liang et al. 2021): two "
                        "independent forward passes of the same clean input (different "
                        "dropout masks each time, model stays in train() mode), "
                        "consistency-regularized via symmetric KL divergence between the "
                        "two output distributions (alpha=1.0). Mechanically distinct from "
                        "everything else tried today -- doesn't perturb inputs (unlike "
                        "FGM/FreeLB) or weights (unlike AWP), doesn't restructure "
                        "representation space (unlike SupCon), just penalizes sensitivity "
                        "to the model's own dropout noise. A sibling project "
                        "(DialectSentEval) found this roughly noise-level (OOF close to "
                        "FGM, real test below it) -- worth an isolated ALEXIS-specific "
                        "test rather than assuming the same outcome transfers.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "rdrop": True,
        "rdrop_alpha": 1.0,
    },

    "exp40_deberta_large_contrastive_v2": {
        "description": "exp09's exact recipe + 15 NEW hand-authored contrastive minimal "
                        "pairs (30 examples) targeting two patterns found by actually "
                        "reading exp09's true-OOF misclassified posts (2026-08-07), "
                        "distinct from exp26's pairs (which targeted violence-direction "
                        "and hypothetical/non-current framing, and were confirmed "
                        "negative): (A) farewell/goodbye-note framing ('this is goodbye', "
                        "'if you're reading this...', saying goodbyes, giving away "
                        "possessions) is a real behavior-level signal (writing a suicide "
                        "note is a recognized preparatory action) that exp09 systematically "
                        "under-weighs, reading it as plain sad/ideation language instead -- "
                        "55 behavior posts were misclassified as ideation, several with this "
                        "exact farewell framing at >90% confidence; (B) the reverse error: "
                        "exp09 over-triggers 'behavior' on posts that only mention a method "
                        "hypothetically/generally ('what method do people use', reading a "
                        "forum) without concrete personal timing/acquisition, which should "
                        "stay 'ideation' -- 67 ideation posts were misclassified as behavior, "
                        "several mentioning methods only in passing. Same isolation "
                        "discipline as exp26: only this one data source changes.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "oversample_contrastive_v2",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp41_deberta_large_multisample_dropout": {
        "description": "exp09's exact recipe + multi-sample dropout (Inoue 2019, k=4): "
                        "one backbone forward pass (unchanged), but 4 independent dropout "
                        "masks applied to the pooled representation before the "
                        "classification head, averaging the 4 resulting per-sample losses. "
                        "Cheaper than R-Drop (which reruns the whole backbone twice) since "
                        "only the small head is repeated -- chosen specifically to fit "
                        "alongside other users' jobs on a memory-constrained shared GPU. "
                        "Different mechanism from everything tried today: doesn't perturb "
                        "inputs or weights, doesn't restructure representation space, just "
                        "averages over the head's own dropout noise within a single step.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
        "multi_sample_dropout": True,
        "msd_k": 4,
    },

    "exp42_deberta_large_long_context_clean": {
        "description": "exp09's exact recipe (lr=2e-5, linear warmup, label smoothing "
                        "0.05, patience=4) + max_length 512 -> 768. NOT the same as the "
                        "old exp12_deberta_large_long_context entry above, which reverted "
                        "to a stale lr=1e-5/cosine_warmup recipe that predates exp09's "
                        "LR finding and was never actually run (no results/ dir exists "
                        "for it) -- this isolates long-context as a single clean variable "
                        "against exp09, same discipline as exp32-41. Motivated by a true-"
                        "OOF error analysis on exp09 (2026-08-18): 31/1635 posts (1.9%) "
                        "exceed 512 tokens and get truncated at inference, and those 31 "
                        "posts score 0.645 accuracy vs 0.803 for the rest (-15.8pts). The "
                        "'attempt' class (already the weakest, F1=0.73) also averages the "
                        "longest posts of all 4 classes (145.8 tokens vs indicator's 58.9) "
                        "-- so this most directly helps the class that most needs it. "
                        "batch_size halved + grad_accum doubled vs exp09 to keep the same "
                        "effective batch size under the extra memory cost of longer "
                        "sequences.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "weighted_ce",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "label_smoothing": 0.05,
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 8,
        "grad_accum": 2,
        "patience": 4,
        "max_length": 768,
    },

    "exp43_deberta_soft_f1": {
        "description": "exp09's exact recipe, only the loss changed: weighted CE -> "
                        "Macro Double Soft-F1 (see trainer.py's MacroDoubleSoftF1Loss). "
                        "Sourced from the IEEE BigData 2024 Cup's winning suicide-detection "
                        "approach (a prior edition of this same competition), which used "
                        "this exact loss instead of any cross-entropy variant. Every loss "
                        "tried in this project so far (weighted CE, label smoothing, focal, "
                        "the CORAL blend) is a CE-family proxy for the metric we actually "
                        "care about; this directly optimizes a differentiable F1 "
                        "approximation instead -- a different training objective, not "
                        "another CE variant, isolated as the only variable against exp09.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "soft_f1",
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },

    "exp44_deberta_cdw_ce": {
        "description": "exp09's exact recipe, only the loss changed: weighted CE -> "
                        "Class Distance Weighted Cross-Entropy (CDW-CE, Polat et al. "
                        "2022, arXiv 2202.05167). A structurally different ordinal "
                        "mechanism from CORAL (already tried, 0.7958 solo -- a "
                        "near-miss vs exp09's 0.8003): CORAL adds an auxiliary head; "
                        "CDW-CE changes nothing about the model, only the loss -- "
                        "penalizes probability mass on wrong classes in proportion "
                        "to (ordinal distance)^2, extended with this project's "
                        "standard class weights. Directly motivated by exp09's own "
                        "error analysis: errors concentrate overwhelmingly at "
                        "adjacent ordinal boundaries (indicator-ideation: 113, "
                        "ideation-behavior: 122, behavior-attempt: 39) vs rare "
                        "non-adjacent jumps (indicator-attempt: 12) -- the model "
                        "already has ordinal structure, this loss is a second, "
                        "cheaper attempt (vs CORAL) to sharpen it further.",
        "model_name": "microsoft/deberta-v3-large",
        "data": "original",
        "loss": "cdw_ce",
        "cdw_alpha": 2.0,
        "class_weights": [0.669, 0.788, 1.045, 3.586],
        "lr": 2e-5,
        "scheduler": "linear_warmup",
        "warmup_ratio": 0.06,
        "epochs": 32,
        "batch_size": 16,
        "grad_accum": 1,
        "patience": 4,
        "max_length": 512,
    },
}

# Dry-run config (tiny, just to test pipeline)
DRY_RUN = {
    "description": "Dry run — 2 folds, 2 epochs, tiny batch, verify pipeline works",
    "model_name": "microsoft/deberta-v3-base",
    "data": "original",
    "loss": "weighted_ce",
    "class_weights": [0.669, 0.788, 1.045, 3.586],
    "label_smoothing": 0.05,
    "lr": 2e-5,
    "scheduler": "cosine_warmup",
    "warmup_ratio": 0.1,
    "epochs": 2,
    "batch_size": 4,
    "grad_accum": 1,
    "patience": 99,
    "max_length": 128,
    "_folds": 2,
}
