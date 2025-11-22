import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

class CubiCasaParser:
    def __init__(self, data_dir="data/raw/cubicasa5k"):
        self.data_dir = data_dir

    def get_svg_dimensions(self, root):
        """Extracts dimensions from SVG viewBox or width/height attributes."""
        width, height = 0, 0
        
        # Try viewBox first (standard for CubiCasa)
        # viewBox format: "min_x min_y width height"
        viewbox = root.get('viewBox')
        if viewbox:
            parts = viewbox.replace(',', ' ').split()
            if len(parts) == 4:
                width = float(parts[2])
                height = float(parts[3])
        
        # Fallback to attributes
        if width == 0 or height == 0:
            try:
                width = float(root.get('width', 0))
                height = float(root.get('height', 0))
            except:
                pass
                
        return width, height

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
            return [x_min, y_min, x_max, y_max] # Return floats for scaling
        except Exception:
            return None

    def strip_namespace(self, tag):
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag

    def is_uuid(self, label):
        if not label: return True
        if len(label) > 30 and "-" in label: return True
        if label.lower() in ["space", "fixedfurniture", "null", "none", "undefined", "ulkotila"]: return True
        return False

    def process_sample(self, svg_path):
        extracted_data = []
        sample_id = os.path.basename(os.path.dirname(svg_path))
        folder = os.path.dirname(svg_path)
        
        # 1. Identify Image Path
        img_name = "F1_scaled.png"
        img_path = os.path.join(folder, img_name)
        if not os.path.exists(img_path):
            if os.path.exists(os.path.join(folder, "F1_original.png")):
                img_path = os.path.join(folder, "F1_original.png")
            else:
                return [] # Skip if no image found

        # 2. Get Image Dimensions (Target Scale)
        try:
            with Image.open(img_path) as img:
                target_w, target_h = img.size
        except:
            return []

        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # 3. Get SVG Dimensions (Source Scale)
            svg_w, svg_h = self.get_svg_dimensions(root)
            if svg_w == 0 or svg_h == 0:
                return [] # Cannot scale without dimensions
            
            # Calculate Scale Factors
            scale_x = target_w / svg_w
            scale_y = target_h / svg_h

            for elem in root.iter():
                tag = self.strip_namespace(elem.tag)
                
                if tag == 'g':
                    elem_class = elem.get("class", "")
                    label = None
                    
                    # Label Extraction Logic
                    if "FixedFurniture" in elem_class:
                        parts = elem_class.split()
                        for p in parts:
                            if p != "FixedFurniture":
                                label = p
                                break
                    
                    if not label:
                        for child in elem.iter():
                            child_tag = self.strip_namespace(child.tag)
                            if child_tag in ['title', 'text', 'tspan']:
                                if child.text and len(child.text.strip()) > 2:
                                    label = child.text.strip()
                                    break
                    
                    if label and not self.is_uuid(label):
                        bbox_raw = None
                        for child in elem:
                            child_tag = self.strip_namespace(child.tag)
                            if child_tag == 'polygon':
                                bbox_raw = self.parse_polygon_str(child.get("points"))
                                break
                            elif child_tag == 'rect':
                                x = float(child.get('x', 0))
                                y = float(child.get('y', 0))
                                w = float(child.get('width', 0))
                                h = float(child.get('height', 0))
                                bbox_raw = [x, y, x+w, y+h]
                                break
                        
                        if bbox_raw:
                            # 4. Apply Scaling
                            x1, y1, x2, y2 = bbox_raw
                            nx1 = int(x1 * scale_x)
                            ny1 = int(y1 * scale_y)
                            nx2 = int(x2 * scale_x)
                            ny2 = int(y2 * scale_y)
                            
                            # Clamp to image boundaries to prevent crashes
                            nx1 = max(0, min(nx1, target_w))
                            ny1 = max(0, min(ny1, target_h))
                            nx2 = max(0, min(nx2, target_w))
                            ny2 = max(0, min(ny2, target_h))
                            
                            # Ensure valid box
                            if nx2 > nx1 and ny2 > ny1:
                                extracted_data.append({
                                    "sample_id": sample_id,
                                    "image_path": img_path,
                                    "label": label,
                                    "bbox": str([nx1, ny1, nx2, ny2])
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
        
        print(f"Found {len(svg_files)} floor plans. Parsing & Rescaling...")
        
        for svg_file in tqdm(svg_files):
            records = self.process_sample(svg_file)
            all_records.extend(records)
            
        df = pd.DataFrame(all_records)
        if len(df) > 0:
            df.drop_duplicates(subset=['sample_id', 'label', 'bbox'], inplace=True)
            output_path = "data/processed/parsed_layout.csv"
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {len(df)} rescale-corrected annotations to {output_path}")
            print(df.head())
        else:
            print("⚠️ No annotations found.")

if __name__ == "__main__":
    parser = CubiCasaParser(data_dir="data/raw") 
    parser.run()