"""Stage 1 of the cross-encoder/entailment idea: supervised fine-tuning of
the same off-the-shelf NLI checkpoint used in Stage 0 (nli_predict.py,
zero-shot honest score 0.288 -- too weak alone, but a real starting point,
unlike training an encoder from scratch which already failed once this
project, v14's dual-encoder, OOF 0.38).

Uses AutoModelForSequenceClassification's existing 3-way NLI head
(entailment/neutral/contradiction) rather than discarding it for a fresh
binary head -- entailment is index 0 for this checkpoint (verified). We
train toward our binary factor-presence label via BCEWithLogitsLoss on
just that one logit, keeping the model in the same "premise entails
hypothesis" framing it was pretrained for, rather than repurposing its
pooled representation for an unrelated fresh task.

Supports freeze_backbone()/unfreeze_backbone() for LP-FT (Kumar et al.,
ICLR 2022, "Fine-Tuning can Distort Pretrained Features and Underperform
Out-of-Distribution") -- train the head first with the backbone frozen,
then unfreeze and continue at a low LR. Their finding: this recovers most
of full fine-tuning's benefit while avoiding most of the OOD damage, which
matters here given we have a demonstrated train-vs-leaderboard shift
(adversarial validation AUC~0.69) and very little data (1,635 posts) to
fine-tune a 435M-parameter model safely.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ENTAILMENT_IDX = 0  # verified: cfg.id2label == {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


class NLIFactorClassifier(nn.Module):
    def __init__(self, model_name=NLI_MODEL):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        return out.logits[:, ENTAILMENT_IDX]  # (B,) -- one logit per pair

    def _backbone_module(self):
        # DebertaV2ForSequenceClassification's encoder submodule name
        return self.model.deberta

    def freeze_backbone(self):
        for p in self._backbone_module().parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self._backbone_module().parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
