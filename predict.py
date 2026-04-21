import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import SkinLesionModel
import os

def predict(image_path, model_path='models/skin_lesion_final.pth', num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class names based on directory names
    class_names = [
        "Eczema", "Warts/Molluscum", "Melanoma", "Atopic Dermatitis", 
        "BCC", "NV", "BKL", "Psoriasis", "Seborrheic Keratoses", "Tinea/Fungal"
    ]
    
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return
    
    model = SkinLesionModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Preprocess Image
    input_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()
        confidence = probabilities[pred_idx].item()
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
        
    print(f"Result: {pred_label} (Confidence: {confidence:.2%})")
    print("-" * 30)
    for i, name in enumerate(class_names):
        print(f"{name:<25}: {probabilities[i].item():.2%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict skin disease from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default="models/skin_lesion_final.pth", help="Path to the model file")
    args = parser.parse_args()
    
    predict(args.image, args.model)
