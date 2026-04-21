import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from model import SkinLesionModel
from preprocess import preprocess_image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = "models/skin_lesion_final.pth"
class_names = [
    "Dry, Itchy Skin (Eczema)", 
    "Warts or Viral Bumps", 
    "Dangerous Skin Cancer (Melanoma)", 
    "Severe Eczema (Atopic Dermatitis)", 
    "Common Skin Cancer (BCC)", 
    "Normal Mole", 
    "Harmless Age Spots", 
    "Scaly Skin Patches (Psoriasis)", 
    "Harmless Skin Growths", 
    "Fungal Infection"
]

DISEASE_INFO = {
    "Dry, Itchy Skin (Eczema)": {
        "causes": "Your skin is very sensitive and reacts strongly to things like stress, dry air, certain soaps, or even the clothes you wear.",
        "precautions": ["Put thick creams or ointments on your skin every day, especially right after a bath.", "Do not use strong soaps or detergents that can dry out your skin.", "Stay away from rough clothes like wool that make you itch.", "Use a machine that puts moisture in the air (humidifier) if your room is very dry.", "See a doctor for special medicines if the itching doesn't stop."],
        "images": ["skin_precaution_eczema_1773393275360.png", "cotton.png"]
    },
    "Warts or Viral Bumps": {
        "causes": "A tiny bug (virus) gets into the top layer of your skin. It can spread if you touch the bump and then touch another part of your body.",
        "precautions": ["Do not pick, scratch, or shave near the bump because it will spread to other areas.", "Keep the bump clean, dry, and covered with a plaster/bandage.", "Do not share towels, clothes, or razors with anyone else.", "Wear slippers or shoes in public places where the floor is wet.", "Go to a doctor if the bumps do not go away or if they hurt."],
        "images": ["hands.png", "skin_precaution_hygiene_1773393323728.png"]
    },
    "Dangerous Skin Cancer (Melanoma)": {
        "causes": "Too much sunlight or using tanning beds has badly damaged the skin. This is serious and needs checking quickly.",
        "precautions": ["Go to a doctor or skin specialist IMMEDIATELY to have it checked.", "Check your skin every month for newly shaped or growing dark spots.", "Put good sunscreen on your skin every day when going outside in the sun.", "Never use sun tanning beds.", "Wear hats, long sleeves, and sunglasses to stay out of strong sunlight."],
        "images": ["sunhat.png", "skin_precaution_sunscreen_1773393301481.png"]
    },
    "Severe Eczema (Atopic Dermatitis)": {
        "causes": "This is a type of Eczema that runs in families. Your skin is easily damaged and doesn't hold water well, making it very dry and itchy.",
        "precautions": ["Find out what makes you itch (like dust, pets, or certain foods) and stay away from them.", "Put gentle, heavy creams on your skin right after you wash.", "Take short showers with slightly warm water, never hot water.", "Wear loose cotton clothes so your skin doesn't get irritated.", "Talk to a doctor if you can't stop scratching."],
        "images": ["skin_precaution_eczema_1773393275360.png", "cotton.png"]
    },
    "Common Skin Cancer (BCC)": {
        "causes": "Your skin has been hit by the sun's rays for too long over many years.",
        "precautions": ["Make an appointment to see a skin doctor soon so they can remove it safely.", "Stay out of direct sunlight, especially during the middle of the day. ", "Always use sunscreen before going outside.", "Keep looking at your skin to see if any spot starts growing or bleeding.", "Wear clothes that cover your arms and legs from the sun."],
        "images": ["sunhat.png", "skin_precaution_sunscreen_1773393301481.png"]
    },
    "Normal Mole": {
        "causes": "This is just a normal mole. Most people have them, and they are usually safe.",
        "precautions": ["Check the mole from time to time to see if it changes size, changes color, or starts bleeding.", "Keep the mole safe from sunburns.", "Do not scratch, pick, or cut the mole.", "If the mole ever starts itching or bleeding, see a doctor right away.", "Have a doctor look at all your spots once a year to be safe."],
        "images": ["skin_precaution_sunscreen_1773393301481.png", "sunhat.png"]
    },
    "Harmless Age Spots": {
        "causes": "These are spots that come from getting older or getting a lot of sun. They look like stuck-on scabs or dark spots, but they are not dangerous.",
        "precautions": ["Even though it looks scary, a doctor can check it to confirm it's completely safe.", "Never try to scratch or pick the spot off, because you might get a bad infection or scar.", "Use sunscreen so you don't get more spots like this.", "Keep your skin nice and soft by drinking water and using creams.", "If your clothes rub the spot and make it hurt, a doctor can take it off for you."],
        "images": ["skin_precaution_hygiene_1773393323728.png", "skin_precaution_sunscreen_1773393301481.png"]
    },
    "Scaly Skin Patches (Psoriasis)": {
        "causes": "Your body makes skin cells way faster than normal, causing thick, red, itchy patches to build up on your skin.",
        "precautions": ["Use the creams or medicines that the doctor gives you exactly as they say.", "Keep the skin soft with strong creams so it doesn't crack open and hurt.", "Try to stress less and avoid alcohol or smoking, as these make it much worse.", "A little bit of gentle morning sunlight helps, but do not get a sunburn.", "Talk to people who understand, as having bad skin can make you sad or frustrated."],
        "images": ["skin_precaution_eczema_1773393275360.png"]
    },
    "Harmless Skin Growths": {
        "causes": "We don't know exactly why these spots appear, but they are very common as people get older and look like rough, waxy scabs stuck to the skin.",
        "precautions": ["Leave the spots alone. Do not pick or scratch them off at home.", "If a spot turns very black, bleeds, or itches badly, show it to a doctor to be safe.", "Wear soft clothes that will not catch on the rough spots.", "Use simple soaps that do not burn the spots.", "If you hate how they look or feel, a doctor can safely freeze or scrape them off."],
        "images": ["skin_precaution_sunscreen_1773393301481.png", "skin_precaution_hygiene_1773393323728.png"]
    },
    "Fungal Infection": {
        "causes": "A type of germ called a fungus is growing on your skin. These germs love dark, wet, and warm places like sweaty feet or the groin.",
        "precautions": ["Put the special fungus killing cream exactly where the package says, even after the spot disappears, so it doesn't return.", "Keep your skin very clean and perfectly dry.", "Change your socks and underclothes every single day, or when they get sweaty.", "Don't walk with bare feet in wet public places, like swimming pools or public bathrooms.", "Wash your towels and clothes in hot water so the germs die."],
        "images": ["hands.png", "cotton.png", "skin_precaution_hygiene_1773393323728.png"]
    }
}

model = SkinLesionModel(num_classes=len(class_names))
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_gradcam(input_tensor, target_category):
    target_layers = [model.backbone.layer4[-1] if hasattr(model.backbone, 'layer4') else model.backbone.blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess for prediction
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0]
            
        # Results
        top_prob, top_idx = torch.max(probabilities, dim=0)
        
        # Generate Grad-CAM heatmap
        # We need gradients so we enable them temporarily
        model.zero_grad()
        grayscale_cam = get_gradcam(input_tensor, top_idx.item())
        
        # Overlay heatmap
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Convert images to base64 for frontend
        _, buffer = cv2.imencode('.png', cv2.cvtColor((visualization * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        probs_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        pred_label = class_names[top_idx.item()]
        info = DISEASE_INFO.get(pred_label, {})

        return {
            "prediction": pred_label,
            "confidence": float(top_prob),
            "all_probabilities": probs_dict,
            "heatmap": f"data:image/png;base64,{heatmap_base64}",
            "details": {
                "causes": info.get("causes", "Unknown"),
                "precautions": info.get("precautions", []),
                "precaution_images": info.get("images", [])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
