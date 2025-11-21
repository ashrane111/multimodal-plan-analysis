import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification

class LayoutLMv3Model(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # We use the pretrained 'base' model from Microsoft
        # This aligns with "Fine-tuned LayoutLMv3" from your resume
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=num_labels
        )

    def forward(self, pixel_values, input_ids, attention_mask, bbox, labels=None):
        # Pass inputs to the HF model
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            labels=labels
        )