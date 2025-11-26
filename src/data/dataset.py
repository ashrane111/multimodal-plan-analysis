import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
import numpy as np
from datasets import load_from_disk
import re

Image.MAX_IMAGE_PIXELS = None 

def map_label_to_ontology(raw_label):
    text = str(raw_label).lower().strip()
    text_clean = re.sub(r'[+/\.]', ' ', text)
    
    # 1. Filter Dimensions
    if re.search(r'\d+[\'\"xX]\d+', text) or re.search(r'\d+\s*sq', text):
        return "Background"
    
    # 2. MAPPING LOGIC
    
    # --- BEDROOM ---
    if "makuu" in text or "alkovi" in text or "bedroom" in text: return "Bedroom"
    if re.search(r'\bmh\s*\d*', text_clean): return "Bedroom" 
    if re.search(r'\b(bed|guest|suite|dorm|bunk)\b', text_clean): return "Bedroom"

    # --- KITCHEN ---
    if "keitti" in text or "kitchen" in text or "cooking" in text: return "Kitchen"
    if re.search(r'\b(k|kt|ks|rt|ruok|dining|pantry|cook|nook|apuk|avok|pk)\b', text_clean): return "Kitchen"

    # --- DINING ---
    if "ruokailu" in text or "dining" in text: return "Dining"

    # --- BATHROOM ---
    if "kylpy" in text or "pesu" in text or "sauna" in text or "bath" in text or "toilet" in text: return "Bathroom"
    if re.search(r'\b(wc|kph|ph|sh|psh|kh|s|shower|restroom|powder)\b', text_clean): return "Bathroom"
    if "puku" in text or "pkh" in text: return "Bathroom"

    # --- LIVING ROOM ---
    if "olohuone" in text or "living" in text or "lounge" in text or "tupa" in text: return "LivingRoom"
    if "oleskelu" in text: return "LivingRoom"
    if re.search(r'\b(oh|family|great|media|rec|ask|ark|salon|takka)\b', text_clean): return "LivingRoom"

    # --- UTILITY --- 
    if re.search(r'\b(utility|laundry|khh|tekn|pannu|pannuh|ljh|öljy|boiler|mud)\b', text_clean): return "Utility"

    # --- STORAGE ---
    if "varasto" in text or "vaate" in text or "storage" in text or "closet" in text: return "Storage"
    if any(x in text for x in ["kellari", "kylmiö", "komero", "ullakko", "katt"]): return "Storage"
    if re.search(r'\b(vh|sk|kom|var|walk-in|wic|wardrobe|khh|lämm|kuiv|kell|kylm|kyl|kylmä|puuvar)\b', text_clean): return "Storage"

    # --- OUTDOOR ---
    if "parveke" in text or "terassi" in text or "veranta" in text or "kuisti" in text: return "Outdoor"
    if re.search(r'\b(out|deck|porch|terrace|balcony|patio|piha|pergola|vilpola|lasitettu)\b', text_clean): return "Outdoor"

    # --- GARAGE ---
    if "auto" in text or "garage" in text: return "Garage"
    if re.search(r'\b(car|parking|at|katos|vaja|tall)\b', text_clean): return "Garage"

    # --- ENTRY ---
    if "eteinen" in text or "entry" in text or "hall" in text or "aula" in text: return "Entry"
    if "kura" in text: return "Entry"
    if re.search(r'\b(et|tk|lobby|foyer|tuuli|pr|tkh)\b', text_clean): return "Entry"

    # --- HALLWAY ---
    if "käytävä" in text or "corridor" in text: return "Hallway"

    # --- OFFICE ---
    if "työ" in text or "office" in text or "study" in text: return "Office"
    if re.search(r'\b(den|library|work|kirjasto|tyoh)\b', text_clean): return "Office"
    
    # --- STAIRS ---
    if "parvi" in text or "loft" in text: return "Bedroom" 
    if "porras" in text or "stair" in text: return "Stairs"
    
    return "Background"

class FloorPlanDataset(Dataset):
    def __init__(self, arrow_path, processor=None):
        full_dataset = load_from_disk(arrow_path)
        self.processor = processor
        
        # 14 Standard Classes
        self.labels = [
            "Background", "LivingRoom", "Kitchen", "Bedroom", "Bathroom", 
            "Entry", "Hallway", "Outdoor", "Storage", "Garage", "Office", 
            "Stairs", "Dining", "Utility"
        ]
        self.labels.sort()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Filter Logic
        print("Filtering dataset to remove empty images...")
        valid_indices = []
        for i in range(len(full_dataset)):
            raw_labels = full_dataset[i]['label'] 
            has_valid_room = False
            for lbl in raw_labels:
                if map_label_to_ontology(lbl) != "Background":
                    has_valid_room = True
                    break
            if has_valid_room:
                valid_indices.append(i)
        
        self.dataset = full_dataset.select(valid_indices)
        print(f"Filtered: Kept {len(self.dataset)} valid images.")
        
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
        item = self.dataset[idx]
        image = item['image'].convert("RGB")
        raw_labels = item['label']
        bbox_list = item['bbox']
        width, height = image.size
        
        processed_boxes = []
        processed_labels = []
        processed_words = [] # <--- NEW: Track the text strings
        
        for raw_label, bbox_str in zip(raw_labels, bbox_list):
            # 1. Parse Box
            try:
                raw_box = eval(bbox_str) if isinstance(bbox_str, str) else bbox_str
                if not isinstance(raw_box, (list, tuple)): continue
            except:
                continue
                
            box = self.normalize_box(raw_box, width, height)
            
            # 2. Map Label
            clean_label = map_label_to_ontology(raw_label)
            
            if clean_label == "Background":
                label_id = -100
            elif clean_label in self.label2id:
                label_id = self.label2id[clean_label]
            else:
                label_id = -100
            
            # 3. Store
            processed_boxes.append(box)
            processed_labels.append(label_id)
            # Pass the RAW TEXT (string) to the model so it learns "k" -> Kitchen
            processed_words.append(str(raw_label)) 
            
        # 4. Processor
        encoding = self.processor(
            image,
            processed_words,    # <--- FIX: Pass list of STRINGS, not IDs
            boxes=processed_boxes,
            word_labels=processed_labels, 
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