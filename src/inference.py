import os
import torch
import random
import glob
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import numpy as np
import pandas as pd
import re

Image.MAX_IMAGE_PIXELS = None 

# --- MAPPING LOGIC (Synced with Dataset) ---
def map_label_to_ontology(raw_label):
    text = str(raw_label).lower().strip()
    text_clean = re.sub(r'[+/\.]', ' ', text)
    
    if re.search(r'\d+[\'\"xX]\d+', text) or re.search(r'\d+\s*sq', text):
        return "Background"
    
    if "makuu" in text or "alkovi" in text or "bedroom" in text: return "Bedroom"
    if re.search(r'\bmh\s*\d*', text_clean): return "Bedroom" 
    if re.search(r'\b(bed|guest|suite|dorm|bunk)\b', text_clean): return "Bedroom"

    if "keitti" in text or "kitchen" in text or "cooking" in text: return "Kitchen"
    if re.search(r'\b(k|kt|ks|rt|ruok|dining|pantry|cook|nook|apuk|avok|pk)\b', text_clean): return "Kitchen"

    if "ruokailu" in text or "dining" in text: return "Dining"

    if "kylpy" in text or "pesu" in text or "sauna" in text or "bath" in text or "toilet" in text: return "Bathroom"
    if re.search(r'\b(wc|kph|ph|sh|psh|kh|s|shower|restroom|powder)\b', text_clean): return "Bathroom"
    if "puku" in text or "pkh" in text: return "Bathroom"

    if "olohuone" in text or "living" in text or "lounge" in text or "tupa" in text: return "LivingRoom"
    if "oleskelu" in text: return "LivingRoom"
    if re.search(r'\b(oh|family|great|media|rec|ask|ark|salon|takka)\b', text_clean): return "LivingRoom"

    if re.search(r'\b(utility|laundry|khh|tekn|pannu|pannuh|ljh|öljy|boiler|mud)\b', text_clean): return "Utility"

    if "varasto" in text or "vaate" in text or "storage" in text or "closet" in text: return "Storage"
    if any(x in text for x in ["kellari", "kylmiö", "komero", "ullakko", "katt"]): return "Storage"
    if re.search(r'\b(vh|sk|kom|var|walk-in|wic|wardrobe|khh|lämm|kuiv|kell|kylm|kyl|kylmä|puuvar)\b', text_clean): return "Storage"

    if "parveke" in text or "terassi" in text or "veranta" in text or "kuisti" in text: return "Outdoor"
    if re.search(r'\b(out|deck|porch|terrace|balcony|patio|piha|pergola|vilpola|lasitettu)\b', text_clean): return "Outdoor"

    if "auto" in text or "garage" in text: return "Garage"
    if re.search(r'\b(car|parking|at|katos|vaja|tall)\b', text_clean): return "Garage"

    if "eteinen" in text or "entry" in text or "hall" in text or "aula" in text: return "Entry"
    if "kura" in text: return "Entry"
    if re.search(r'\b(et|tk|lobby|foyer|tuuli|pr|tkh)\b', text_clean): return "Entry"

    if "käytävä" in text or "corridor" in text: return "Hallway"

    if "työ" in text or "office" in text or "study" in text: return "Office"
    if re.search(r'\b(den|library|work|kirjasto|tyoh)\b', text_clean): return "Office"
    
    if "parvi" in text or "loft" in text: return "Bedroom"
    if "porras" in text or "stair" in text: return "Stairs"
    
    return "Background"

def normalize_box(box, width, height):
    return [
        int(np.clip((box[0] / width) * 1000, 0, 1000)),
        int(np.clip((box[1] / height) * 1000, 0, 1000)),
        int(np.clip((box[2] / width) * 1000, 0, 1000)),
        int(np.clip((box[3] / height) * 1000, 0, 1000))
    ]

def unnormalize_box(box, width, height):
    return [
        int((box[0] / 1000) * width),
        int((box[1] / 1000) * height),
        int((box[2] / 1000) * width),
        int((box[3] / 1000) * height)
    ]

def get_random_cubicasa_sample(data_root="data/raw/cubicasa5k"):
    all_pngs = []
    print("Searching for a random floor plan...")
    count = 0
    for root, dirs, files in os.walk(data_root):
        if "F1_scaled.png" in files:
            all_pngs.append(os.path.join(root, "F1_scaled.png"))
            count += 1
            if count > 500: break 
    if not all_pngs: raise FileNotFoundError("No F1_scaled.png found!")
    return random.choice(all_pngs)

def run_inference():
    model_path = "models_output/layoutlmv3_finetuned_full"
    print(f"Loading model from {model_path}...")
    
    try:
        processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    id2label = model.config.id2label
    
    max_retries = 10
    for _ in range(max_retries):
        image_path = get_random_cubicasa_sample()
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        df = pd.read_csv("data/processed/parsed_layout.csv")
        sample_id = os.path.basename(os.path.dirname(image_path))
        
        rows = df[df['sample_id'].astype(str) == str(sample_id)]
        if len(rows) == 0: continue

        words = []
        boxes = []
        
        for _, row in rows.iterrows():
            label_text = str(row['label'])
            clean_label = map_label_to_ontology(label_text)
            
            # NOTE: We KEEP background items in the input list so we can see if the model
            # correctly identifies them as background, or if it mistakes Kitchen for Background.
            if clean_label == "Background": continue  #<-- DISABLED FOR DEBUG
                
            raw_box = eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
            
            # Pass RAW TEXT to the model (e.g. "K")
            words.append(label_text) 
            boxes.append(normalize_box(raw_box, width, height))
        
        if len(words) > 0:
            print(f"Testing on: {image_path}")
            break
    else:
        print("Could not find a good sample.")
        return

    # Predict
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**encoding)
        
    # ALIGNMENT LOGIC
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    if isinstance(predictions, int): predictions = [predictions]
    
    token_boxes = encoding.bbox.squeeze().tolist()
    word_ids = encoding.word_ids()
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()

    print("\n--- Predictions ---")
    
    seen_words = set()
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None: continue
        if word_id in seen_words: continue
        
        seen_words.add(word_id)
        
        pred_id = predictions[idx]
        predicted_label = id2label[pred_id]
        original_text = words[word_id]
        original_box = boxes[word_id]
        
        # Calculate Confidence
        probs = torch.softmax(logits[0][idx], dim=0)
        conf = probs[pred_id].item()

        # Visuals
        color = "red"
        if predicted_label == "Bedroom": color = "blue"
        if predicted_label == "Kitchen": color = "green"
        if predicted_label == "Bathroom": color = "cyan"
        if predicted_label == "LivingRoom": color = "orange"
        if predicted_label == "Background": color = "gray" # Grey for background
        
        draw_box = unnormalize_box(original_box, width, height)
        draw.rectangle(draw_box, outline=color, width=5)
        
        # Always print label (Even Background)
        text = f"{predicted_label}"
        text_bbox = draw.textbbox((draw_box[0], draw_box[1]), text, font=font)
        draw.rectangle((text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]), fill="white")
        draw.text((draw_box[0], draw_box[1]), text, fill=color, font=font)
        
        mapped_truth = map_label_to_ontology(original_text)
        print(f"Input: '{original_text}' (Truth: {mapped_truth}) -> Pred: {predicted_label} (Conf: {conf:.4f})")

    output_img_path = "models_output/real_prediction_vis.png"
    image.save(output_img_path)
    print(f"\nSaved visualization to {output_img_path}")

if __name__ == "__main__":
    run_inference()