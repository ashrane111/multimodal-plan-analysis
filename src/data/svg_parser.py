import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

class CubiCasaParser:
    def __init__(self, data_dir="data/raw/cubicasa5k"):
        self.data_dir = data_dir

    def parse_polygon_str(self, points_str):
        try:
            if not points_str: return None
            cleaned = points_str.replace(',', ' ').split()
            points = []
            for i in range(0, len(cleaned), 2):
                x = float(cleaned[i])
                y = float(cleaned[i+1])
                points.append([x, y])
            
            points = np.array(points)
            if len(points) < 2: return None
            
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            return [int(x_min), int(y_min), int(x_max), int(y_max)]
        except Exception:
            return None

    def strip_namespace(self, tag):
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag

    def is_uuid(self, label):
        if not label: return True
        if len(label) > 30 and "-" in label: return True
        if label.lower() in ["space", "fixedfurniture", "null", "none"]: return True
        return False

    def process_sample(self, svg_path):
        extracted_data = []
        sample_id = os.path.basename(os.path.dirname(svg_path))
        
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            for elem in root.iter():
                tag = self.strip_namespace(elem.tag)
                
                if tag == 'g':
                    elem_class = elem.get("class", "")
                    label = None
                    
                    # PRIORITY 1: Extract from Class
                    if "FixedFurniture" in elem_class:
                        parts = elem_class.split()
                        for p in parts:
                            if p != "FixedFurniture":
                                label = p
                                break
                    
                    # PRIORITY 2: Look for Text/Title Tags
                    if not label:
                        for child in elem.iter():
                            child_tag = self.strip_namespace(child.tag)
                            if child_tag in ['title', 'text', 'tspan']:
                                if child.text and len(child.text.strip()) > 2:
                                    label = child.text.strip()
                                    break
                    
                    # PRIORITY 3: Validation & Geometry
                    if label and not self.is_uuid(label):
                        bbox = None
                        for child in elem:
                            child_tag = self.strip_namespace(child.tag)
                            if child_tag == 'polygon':
                                bbox = self.parse_polygon_str(child.get("points"))
                                break
                            elif child_tag == 'rect':
                                x = float(child.get('x', 0))
                                y = float(child.get('y', 0))
                                w = float(child.get('width', 0))
                                h = float(child.get('height', 0))
                                bbox = [int(x), int(y), int(x+w), int(y+h)]
                                break
                        
                        if bbox:
                            folder = os.path.dirname(svg_path)
                            img_name = "F1_scaled.png"
                            if not os.path.exists(os.path.join(folder, img_name)):
                                if os.path.exists(os.path.join(folder, "F1_original.png")):
                                    img_name = "F1_original.png"

                            extracted_data.append({
                                "sample_id": sample_id,
                                "image_path": os.path.join(folder, img_name),
                                "label": label,
                                "bbox": str(bbox)  # <--- FIX: Convert list to string for hashing
                            })

            return extracted_data
        except Exception:
            return []

    def run(self):
        all_records = []
        print(f"Scanning {self.data_dir} for model.svg files...")
        
        svg_files = []
        for root, dirs, files in os.walk(self.data_dir):
            if "model.svg" in files:
                svg_files.append(os.path.join(root, "model.svg"))
        
        print(f"Found {len(svg_files)} floor plans. Parsing...")
        
        for svg_file in tqdm(svg_files):
            records = self.process_sample(svg_file)
            all_records.extend(records)
            
        df = pd.DataFrame(all_records)
        
        if len(df) > 0:
            # Now this works because 'bbox' is a string!
            df.drop_duplicates(subset=['sample_id', 'label', 'bbox'], inplace=True)
            
            output_path = "data/processed/parsed_layout.csv"
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Cleaned Data! Saved {len(df)} annotations to {output_path}")
            print(df.head())
        else:
            print("⚠️ Warning: No valid labels found.")

if __name__ == "__main__":
    parser = CubiCasaParser(data_dir="data/raw") 
    parser.run()