import pandas as pd
from datasets import Dataset, Features, Image, Value, Sequence
import os
from tqdm import tqdm

def create_arrow_dataset():
    print("Reading CSV...")
    csv_path = "data/processed/parsed_layout.csv"
    df = pd.read_csv(csv_path)
    
    print("Grouping data by image (this creates the document structure)...")
    # Group by sample_id/image_path so one row = one full floor plan
    # We aggregate labels and bboxes into lists
    grouped = df.groupby(['sample_id', 'image_path']).agg({
        'label': list,
        'bbox': list
    }).reset_index()
    
    print(f"Found {len(grouped)} unique floor plans.")

    # Define Schema: Note 'Sequence' for lists
    features = Features({
        'sample_id': Value('string'),
        'image': Image(),
        'label': Sequence(Value('string')), # List of strings
        'bbox': Sequence(Value('string'))   # List of strings
    })
    
    # Create Dataset
    print("Creating Arrow Dataset...")
    dataset = Dataset.from_dict({
        "sample_id": grouped['sample_id'].astype(str).tolist(),
        "image": grouped['image_path'].tolist(),
        "label": grouped['label'].tolist(),
        "bbox": grouped['bbox'].tolist()
    }, features=features)
    
    save_path = "data/processed/cubicasa_arrow"
    print(f"Saving to {save_path}...")
    dataset.save_to_disk(save_path)
    print("✅ Success! Arrow dataset is now grouped per image.")

if __name__ == "__main__":
    create_arrow_dataset()