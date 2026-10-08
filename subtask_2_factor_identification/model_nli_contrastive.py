"""Idea 2: same NLI cross-encoder as model_nli_finetune.py, but also exposes
the pooled representation (post-pooler, pre-dropout -- exactly the tensor
DebertaV2ForSequenceClassification.forward() feeds into self.dropout then
self.classifier, verified via inspect.getsource) so an auxiliary contrastive
loss can be applied on it during training. The classification pathway
itself (pooler -> dropout -> classifier) is left byte-for-byte identical to
the original model's forward(), just computed manually here so both the
logit and the pre-dropout pooled tensor can be returned together.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
ENTAILMENT_IDX = 0  # verified: cfg.id2label == {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


class NLIContrastiveClassifier(nn.Module):
    def __init__(self, model_name=NLI_MODEL):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_embedding=False):
        outputs = self.model.deberta(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, return_dict=True,
        )
        encoder_layer = outputs[0]
        pooled_output = self.model.pooler(encoder_layer)          # (B, H) -- contrastive embedding
        logits = self.model.classifier(self.model.dropout(pooled_output))
        entailment_logit = logits[:, ENTAILMENT_IDX]               # (B,)
        if return_embedding:
            return entailment_logit, pooled_output
        return entailment_logit

    def _backbone_module(self):
        return self.model.deberta

    def freeze_backbone(self):
        for p in self._backbone_module().parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self._backbone_module().parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
