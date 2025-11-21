import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

class CubiCasaParser:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir

    def parse_polygon_str(self, points_str):
        """
        Converts SVG polygon string 'x1,y1 x2,y2 ...' into a list of coordinates.
        Returns [x_min, y_min, x_max, y_max] (Bounding Box).
        """
        try:
            # Split by space to get pairs, then comma to get x,y
            points = []
            for p in points_str.strip().split(' '):
                x, y = map(float, p.split(','))
                points.append([x, y])
            
            points = np.array(points)
            # Geometric Transformation: Polygon -> Bounding Box
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            return [int(x_min), int(y_min), int(x_max), int(y_max)]
        except Exception as e:
            print(f"Error parsing points: {points_str} | {e}")
            return [0, 0, 0, 0]

    def process_sample(self, sample_id):
        svg_path = os.path.join(self.data_dir, sample_id, "model.svg")
        if not os.path.exists(svg_path):
            print(f"SVG not found for {sample_id}")
            return []

        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Namespaces can be tricky in SVG, stripping them usually helps
        # For this simple parser, we assume standard tags
        
        extracted_data = []

        # Iterate over Spaces (Rooms)
        # In our mock data, we nested them under <g id="spaces">
        for group in root.findall(".//{http://www.w3.org/2000/svg}g"):
            # Check if this group has a 'class' attribute
            elem_class = group.get("class")
            
            # We are interested in 'Space' (Rooms) and 'FixedFurniture' (Objects)
            if elem_class in ["Space", "FixedFurniture"]:
                
                # 1. Get Label (Title)
                title_elem = group.find("{http://www.w3.org/2000/svg}title")
                label = title_elem.text if title_elem is not None else "Unknown"
                
                # 2. Get Geometry (Polygon)
                poly_elem = group.find("{http://www.w3.org/2000/svg}polygon")
                if poly_elem is not None:
                    points_str = poly_elem.get("points")
                    bbox = self.parse_polygon_str(points_str)
                    
                    extracted_data.append({
                        "sample_id": sample_id,
                        "label": label,
                        "bbox": bbox,
                        "category": elem_class
                    })

        return extracted_data

    def run(self):
        all_records = []
        samples = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        print(f"Found {len(samples)} samples to process...")
        
        for sample in samples:
            records = self.process_sample(sample)
            all_records.extend(records)
            
        # Convert to DataFrame
        df = pd.DataFrame(all_records)
        output_path = "data/processed/parsed_layout.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved parsed data to {output_path}")
        print(df.head())

if __name__ == "__main__":
    parser = CubiCasaParser()
    parser.run()