import os
import numpy as np
from PIL import Image, ImageDraw

def create_mock_cubicasa_sample(base_dir="data/raw", sample_id="sample_01"):
    """
    Creates a mock CubiCasa5k data sample with the exact folder structure
    and XML format needed for the parsing pipeline.
    """
    # 1. Setup Directory Structure
    sample_dir = os.path.join(base_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    
    print(f"Creating mock sample in: {sample_dir}")

    # 2. Create a Dummy Floorplan Image (F1_scaled.png)
    # We create a 500x500 white image with some black lines to simulate a plan
    img_size = (800, 800)
    image = Image.new('RGB', img_size, color='white')
    draw = ImageDraw.Draw(image)
    # Draw some random 'walls'
    draw.rectangle([100, 100, 400, 400], outline="black", width=5) # Living Room
    draw.rectangle([400, 100, 700, 400], outline="black", width=5) # Kitchen
    
    img_path = os.path.join(sample_dir, "F1_scaled.png")
    image.save(img_path)
    print(f"  - Created image: {img_path}")

    # 3. Create the SVG (model.svg)
    # This mimics the actual CubiCasa5k XML structure with 'Space' and 'FixedFurniture'
    svg_content = f"""<svg width="{img_size[0]}" height="{img_size[1]}" xmlns="http://www.w3.org/2000/svg">
    <g id="graph">
        <g id="spaces">
            <g id="space_1" class="Space">
                <text x="250" y="250" font-family="Verdana" font-size="20" fill="black">Living Room</text>
                <polygon points="100,100 400,100 400,400 100,400" fill="none" stroke="blue"/>
                <title>LivingRoom</title>
            </g>
            <g id="space_2" class="Space">
                <text x="550" y="250" font-family="Verdana" font-size="20" fill="black">Kitchen</text>
                <polygon points="400,100 700,100 700,400 400,400" fill="none" stroke="blue"/>
                <title>Kitchen</title>
            </g>
        </g>
        
        <g id="walls">
            <g id="wall_1" class="Wall">
                <polygon points="100,100 700,100 700,110 100,110" fill="black"/>
            </g>
            <g id="wall_2" class="Wall">
                <polygon points="100,400 700,400 700,410 100,410" fill="black"/>
            </g>
        </g>

        <g id="items">
            <g id="item_1" class="FixedFurniture">
                <polygon points="450,150 650,150 650,200 450,200" fill="gray"/>
                <title>KitchenCabinet</title>
            </g>
        </g>
    </g>
</svg>
"""
    
    svg_path = os.path.join(sample_dir, "model.svg")
    with open(svg_path, "w") as f:
        f.write(svg_content)
    print(f"  - Created SVG: {svg_path}")

if __name__ == "__main__":
    # Create 3 mock samples
    for i in range(1, 4):
        create_mock_cubicasa_sample(sample_id=f"sample_{i:02d}")
    print("\n✅ Data generation complete. Ready for pipeline processing.")