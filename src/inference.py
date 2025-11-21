import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from src.data.dataset import FloorPlanDataset # Re-using for label mapping

def run_inference(model_path, image_path):
    print(f"Loading model from {model_path}...")
    
    # 1. Load Model & Processor
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    
    # AUTOMATIC FIX: Get the label map directly from the trained model config
    id2label = model.config.id2label
    print(f"Loaded Label Map: {id2label}")
    
    # 2. Prepare Image
    image = Image.open(image_path).convert("RGB")
    
    # Mock boxes (matches sample_01)
    dummy_words = ["LivingRoom", "Kitchen", "KitchenCabinet"]
    dummy_boxes = [[100, 100, 400, 400], [400, 100, 700, 400], [450, 150, 650, 200]]
    
    # 3. Preprocess
    encoding = processor(
        image,
        dummy_words,
        boxes=dummy_boxes,
        return_tensors="pt"
    )
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(**encoding)
        
    # Get predictions (indices)
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    
    # 5. Visualize
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    print("\n--- Inference Results ---")
    
    for box, pred_id, word in zip(dummy_boxes, predictions, dummy_words):
        # Use the model's own internal map to decode
        predicted_label = id2label[pred_id]
        
        # Draw Box
        draw.rectangle(box, outline="red", width=3)
        
        # Draw Text
        text = f"{predicted_label} ({word})"
        draw.text((box[0] + 5, box[1] + 5), text, fill="red", font=font)
        
        print(f"Word: '{word}' | Predicted: {predicted_label}")

    # Save result
    output_img_path = "models_output/prediction_vis.png"
    image.save(output_img_path)
    print(f"\nSaved visualization to {output_img_path}")

if __name__ == "__main__":
    # No need to manually pass labels anymore!
    run_inference(
        model_path="models_output/layoutlmv3_finetuned",
        image_path="data/raw/sample_01/F1_scaled.png"
    )