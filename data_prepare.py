import cv2
import numpy as np
import os
from tqdm import tqdm
from preprocess import remove_hair

def process_dataset(input_dir='data', output_dir='data_processed', size=(224, 224)):
    """
    Applies hair removal and resizing to the entire dataset.
    Saves the cleaned images to a new directory while maintaining the folder structure.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the directories
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directory in output_dir
        relative_path = os.path.relpath(root, input_dir)
        target_root = os.path.join(output_dir, relative_path)
        
        if not os.path.exists(target_root):
            os.makedirs(target_root)

        # Process images
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue

        print(f"Processing folder: {relative_path} ({len(image_files)} images)...")
        for filename in tqdm(image_files):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(target_root, filename)
            
            # Skip if already exists
            if os.path.exists(output_path):
                continue

            try:
                # 1. Load
                image = cv2.imread(input_path)
                if image is None:
                    continue
                
                # 2. Hair removal
                clean_image = remove_hair(image)
                
                # 3. Resize
                resized = cv2.resize(clean_image, size)
                
                # 4. Save
                cv2.imwrite(output_path, resized)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    print(f"\nPreprocessing complete! Cleaned data saved in: {output_dir}")

if __name__ == "__main__":
    process_dataset()
