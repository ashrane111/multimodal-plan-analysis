import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification

class LayoutLMv3Model(nn.Module):
    def __init__(self, num_labels, id2label=None, label2id=None):
        super().__init__()
        # Pass the label maps to the pretrained model so they get saved in config.json
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )

    def forward(self, pixel_values, input_ids, attention_mask, bbox, labels=None):
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            labels=labels
        )