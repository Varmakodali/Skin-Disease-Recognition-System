import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from glob import glob

def remove_hair(image):
    """
    Applies a Black Hat filter to remove hair artifacts from dermoscopic images.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kernel for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Apply Black Hat filter
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Intensify the hair pixels
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image based on the thresholded mask
    inpainted = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    
    return inpainted

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads, removes hair, and resizes an image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Remove hair
    clean_image = remove_hair(image)
    
    # Resize
    resized_image = cv2.resize(clean_image, target_size)
    
    return resized_image

if __name__ == "__main__":
    # Example usage / Test
    # This script assumes images are in 'data/HAM10000_images/'
    # and metadata is in 'data/HAM10000_metadata.csv'
    
    print("Preprocessing script initialized.")
    # In a real scenario, this would loop through the dataset and save processed images or prepare masks.
