import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from tqdm import tqdm

class CubiCasaParser:
    def __init__(self, data_dir="data/raw/cubicasa5k"):
        self.data_dir = data_dir

    def parse_polygon_str(self, points_str):
        """
        Robustly parses polygon strings with various whitespace formats.
        Returns [x_min, y_min, x_max, y_max].
        """
        try:
            if not points_str: return None
            points = []
            # Handle "x,y x,y" or "x,y,x,y" or "x y x y"
            # Replace commas with spaces to normalize
            cleaned = points_str.replace(',', ' ').split()
            # Iterate in pairs
            for i in range(0, len(cleaned), 2):
                x = float(cleaned[i])
                y = float(cleaned[i+1])
                points.append([x, y])
            
            points = np.array(points)
            if len(points) < 2: return None # Need at least a line
            
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            return [int(x_min), int(y_min), int(x_max), int(y_max)]
        except Exception:
            return None

    def strip_namespace(self, tag):
        """Removes the {http://...} namespace prefix from tags."""
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag

    def process_sample(self, svg_path):
        extracted_data = []
        sample_id = os.path.basename(os.path.dirname(svg_path))
        
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Walk through EVERY element in the tree (recursive)
            for elem in root.iter():
                # Clean the tag name (remove namespace)
                tag = self.strip_namespace(elem.tag)
                
                # We are looking for Groups <g> that represent Spaces or Items
                if tag == 'g':
                    # Check class attribute (e.g., class="Space")
                    elem_class = elem.get("class", "")
                    
                    if "Space" in elem_class or "FixedFurniture" in elem_class:
                        # STRATEGY 1: Look for <title> tag inside
                        label = None
                        for child in elem:
                            child_tag = self.strip_namespace(child.tag)
                            if child_tag == 'title':
                                label = child.text
                                break
                            elif child_tag == 'text': # Fallback to text tag
                                label = child.text
                                break
                        
                        # STRATEGY 2: Look for 'label' attribute on the group itself
                        if not label:
                            label = elem.get("label") or elem.get("id")

                        # STRATEGY 3: Look for polygon geometry
                        bbox = None
                        for child in elem:
                            child_tag = self.strip_namespace(child.tag)
                            if child_tag == 'polygon':
                                bbox = self.parse_polygon_str(child.get("points"))
                                break
                            elif child_tag == 'rect':
                                # Handle <rect> geometry just in case
                                x = float(child.get('x', 0))
                                y = float(child.get('y', 0))
                                w = float(child.get('width', 0))
                                h = float(child.get('height', 0))
                                bbox = [int(x), int(y), int(x+w), int(y+h)]
                                break
                        
                        # If we found both a Label and a Box, save it!
                        if label and bbox:
                            # Check for valid image file
                            # Real CubiCasa folders usually have F1_scaled.png or F1_original.png
                            folder = os.path.dirname(svg_path)
                            img_name = "F1_scaled.png"
                            if not os.path.exists(os.path.join(folder, img_name)):
                                # Fallback check
                                if os.path.exists(os.path.join(folder, "F1_original.png")):
                                    img_name = "F1_original.png"

                            extracted_data.append({
                                "sample_id": sample_id,
                                "image_path": os.path.join(folder, img_name),
                                "label": label,
                                "bbox": bbox
                            })

            return extracted_data
        except Exception as e:
            # print(f"Error parsing {sample_id}: {e}") # Optional: Uncomment to debug
            return []

    def run(self):
        all_records = []
        print(f"Scanning {self.data_dir} for model.svg files...")
        
        svg_files = []
        for root, dirs, files in os.walk(self.data_dir):
            if "model.svg" in files:
                svg_files.append(os.path.join(root, "model.svg"))
        
        print(f"Found {len(svg_files)} floor plans. Parsing...")
        
        # Use tqdm to show progress
        for svg_file in tqdm(svg_files):
            records = self.process_sample(svg_file)
            all_records.extend(records)
            
        if len(all_records) > 0:
            df = pd.DataFrame(all_records)
            output_path = "data/processed/parsed_layout.csv"
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Success! Saved {len(df)} annotations to {output_path}")
            print(df.head())
        else:
            print("❌ Still found 0 annotations. Please check the SVG structure manually.")

if __name__ == "__main__":
    # Point this to wherever you unzipped the folder
    # If you unzipped inside data/raw, it might be data/raw/cubicasa5k
    parser = CubiCasaParser(data_dir="data/raw") 
    parser.run()