import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from model import SkinLesionModel
from PIL import Image

def generate_heatmap(model_path, image_path, output_path, num_classes=10):
    """
    Generates a Grad-CAM heatmap for a given image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class names based on directory names in 'data/'
    class_names = [
        "Eczema", "Warts/Molluscum", "Melanoma", "Atopic Dermatitis", 
        "BCC", "NV", "BKL", "Psoriasis", "Seborrheic Keratoses", "Tinea/Fungal"
    ]
    
    # Load model
    model = SkinLesionModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Target layer for Grad-CAM
    # For EfficientNetV2-L in timm, conv_head is a good target
    target_layers = [model.backbone.conv_head]
    
    # Load and preprocess image
    input_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"

    # Create Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate heatmap
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay heatmap on original image
    rgb_img = np.array(input_image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image\nPred: {pred_label} ({confidence:.2%})")
    plt.imshow(rgb_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the source image")
    parser.add_argument("--model", type=str, default="models/skin_lesion_final.pth", help="Path to the trained model weight")
    parser.add_argument("--output", type=str, default="results/grad_cam_output.png", help="Path to save the result")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
    elif not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
    else:
        print(f"Generating heatmap for {args.image}...")
        generate_heatmap(args.model, args.image, args.output)
