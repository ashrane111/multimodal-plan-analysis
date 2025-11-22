import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np

class FloorPlanDataset(Dataset):
    def __init__(self, arrow_path, processor=None):
        # 1. Load the Arrow Dataset (Instantaneous because of Memory Mapping)
        self.dataset = load_from_disk(arrow_path)
        self.processor = processor
        
        # 2. Create Label Mapping
        # We extract unique labels from the arrow dataset
        unique_labels = sorted(list(set(self.dataset['label'])))
        self.labels = unique_labels
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
    def __len__(self):
        return len(self.dataset)

    def normalize_box(self, box, width, height):
        return [
            int(np.clip((box[0] / width) * 1000, 0, 1000)),
            int(np.clip((box[1] / height) * 1000, 0, 1000)),
            int(np.clip((box[2] / width) * 1000, 0, 1000)),
            int(np.clip((box[3] / height) * 1000, 0, 1000))
        ]

    def __getitem__(self, idx):
        # 1. Get the row from Arrow
        # This automatically decodes the image bytes into a PIL object!
        item = self.dataset[idx]
        
        image = item['image'].convert("RGB") # Arrow returns PIL Image automatically
        label_text = item['label']
        bbox_str = item['bbox']
        
        width, height = image.size
        
        # 2. Process Layout
        # Eval string box to list
        raw_box = eval(bbox_str)
        
        # Normalize
        box = self.normalize_box(raw_box, width, height)
        
        # Map Label
        label_id = self.label2id[label_text]
        
        # 3. Processor
        # LayoutLMv3 requires lists of boxes/labels even for single items
        encoding = self.processor(
            image,
            [label_text],       # Processor expects a list of words
            boxes=[box],        # Processor expects a list of boxes
            word_labels=[label_id],
            truncation=True,
            padding="max_length", 
            max_length=512,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "bbox": encoding["bbox"].squeeze(),
            "labels": encoding["labels"].squeeze()
        }