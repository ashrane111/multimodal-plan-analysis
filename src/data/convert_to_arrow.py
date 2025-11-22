import pandas as pd
from datasets import Dataset, Features, Image, Value, Sequence
import os

def create_arrow_dataset():
    # 1. Read the CSV
    csv_path = "data/processed/parsed_layout.csv"
    df = pd.read_csv(csv_path)
    
    # 2. Fix Image Paths (Ensure they are absolute or relative correct)
    # The CSV has paths like 'data/raw/...', which is correct relative to root.
    
    # 3. Define the Schema (Features)
    # This tells Arrow exactly what binary format to expect
    features = Features({
        'image': Image(),                   # Stores the raw image bytes efficiently
        'label': Value('string'),           # The text label
        'bbox': Value('string'),            # The bounding box string
        'sample_id': Value('string')        # The ID
    })
    
    # 4. Create the Dataset
    # We convert the DataFrame into a dictionary format first
    # Note: We do NOT load images yet. Hugging Face 'Image()' feature handles lazy loading.
    dataset = Dataset.from_dict({
        "image": df['image_path'].tolist(), # Pass paths, HF will load them automatically
        "label": df['label'].tolist(),
        "bbox": df['bbox'].tolist(),
        "sample_id": df['sample_id'].tolist()
    }, features=features)
    
    print(f"Created dataset with {len(dataset)} rows.")
    
    # 5. Save to Disk (This creates the Arrow files)
    save_path = "data/processed/cubicasa_arrow"
    dataset.save_to_disk(save_path)
    print(f"✅ Saved Arrow dataset to {save_path}")

if __name__ == "__main__":
    create_arrow_dataset()