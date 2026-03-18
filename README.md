<h1 align="center">🩺 Skin Disease Recognition System</h1>

<p align="center">
  <img src="results/grad_cam_melanoma.png" alt="Grad-CAM Visualization" width="500"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Web%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-EfficientNetV2-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/10%20Classes-Skin%20Diseases-brightgreen?style=for-the-badge" />
</p>

> An intelligent, AI-powered web diagnostic application that recognizes 10 common skin diseases from a single uploaded photo — with visual explanations and easy-to-understand medical advice for everyone.

---

## 🌟 What Makes This Project Special

| Feature | Details |
|---|---|
| 🤖 **AI Model** | EfficientNet-B0 with custom SE Attention Blocks |
| 🔍 **Explainability** | Grad-CAM heatmap shows exactly where the model looked |
| ✅ **Real vs. Fake Detection** | Rejects non-skin or invalid images with confidence threshold |
| 🗣️ **Plain English** | All disease names and medical advice are written in simple, easy-to-understand words |
| 💚 **Beautiful UI** | Full-screen Emerald Green Medical Dashboard |
| 📷 **Live Camera** | Analyze images from your webcam directly |
| ⚡ **Fast Inference** | Runs on both CPU and GPU |

---

## 🔬 Recognized Skin Conditions

The model classifies **10 diseases** from a curated dataset of over **25,000 dermoscopic images**:

| # | Class Name (Technical) | App Display Name |
|---|---|---|
| 1 | Eczema | Dry, Itchy Skin (Eczema) |
| 2 | Warts/Molluscum | Warts or Viral Bumps |
| 3 | Melanoma | Dangerous Skin Cancer (Melanoma) |
| 4 | Atopic Dermatitis | Severe Eczema (Atopic Dermatitis) |
| 5 | Basal Cell Carcinoma (BCC) | Common Skin Cancer (BCC) |
| 6 | Melanocytic Nevi (NV) | Normal Mole |
| 7 | Benign Keratosis-like Lesions (BKL) | Harmless Age Spots |
| 8 | Psoriasis | Scaly Skin Patches (Psoriasis) |
| 9 | Seborrheic Keratoses | Harmless Skin Growths |
| 10 | Tinea/Fungal Infection | Fungal Infection |

---

## 📁 Project Structure

```text
skin-disease-recognition-system/
│
├── data/                          # 📦 Raw dataset (10 class folders, ~25k images)
│   ├── 1. Eczema 1677/
│   ├── 2. Melanoma 15.75k/
│   ├── 3. Atopic Dermatitis - 1.25k/
│   ├── 4. Basal Cell Carcinoma (BCC) 3323/
│   ├── 5. Melanocytic Nevi (NV) - 7970/
│   ├── 6. Benign Keratosis-like Lesions (BKL) 2624/
│   ├── 7. Psoriasis pictures - 2k/
│   ├── 8. Seborrheic Keratoses - 1.8k/
│   ├── 9. Tinea Ringworm - 1.7k/
│   └── 10. Warts Molluscum - 2103/
│
├── data_processed/                # 🧹 Cleaned dataset after hair removal + resize
│
├── models/                        # 🧠 Saved model weights (*.pth)
│   ├── skin_lesion_final.pth
│   └── skin_lesion_model_epoch_5.pth
│
├── notebooks/                     # 📒 Jupyter notebooks for exploration
│
├── results/                       # 📊 Training logs & Grad-CAM images
│   ├── training_log.csv
│   ├── grad_cam_melanoma.png
│   ├── grad_cam_atopic.png
│   └── model_performance.png
│
├── src/                           # 💻 Core Python source code
│   ├── api.py          # FastAPI server — main entry point for the web app
│   ├── model.py        # EfficientNet + SE Attention model architecture & Focal Loss
│   ├── train.py        # Full training pipeline (80/20 train-val split)
│   ├── preprocess.py   # Black-Hat morphological hair removal filter
│   ├── data_prepare.py # Batch preprocessing & dataset cleaning
│   ├── explain.py      # CLI Grad-CAM heatmap generation tool
│   ├── predict.py      # CLI tool for single-image prediction
│   └── plot_results.py # Training curve visualization
│
├── static/                        # 🌐 Frontend Web Dashboard
│   ├── index.html      # Main single-page app structure
│   ├── style.css       # Emerald green glassmorphism theme
│   └── script.js       # UI logic, camera, and API integration
│
├── requirements.txt               # 📦 All Python dependencies
├── .gitignore                     # 🚫 Files excluded from Git
└── README.md                      # 📖 This file
```

---

## 🚀 Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/skin-disease-recognition-system.git
cd skin-disease-recognition-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
> ✅ Requires **Python 3.10+**. A GPU is highly recommended for training.

### Step 3: Add Your Dataset
Download the dataset from [Kaggle - Skin Diseases Image Dataset](https://www.kaggle.com/) and place the disease folders inside the `data/` directory.

---

## ▶️ Running the Application

### 🌐 Launch the Web Dashboard
```bash
python src/api.py
```
Then open your browser at **http://localhost:8000**

---

## 🛠️ Training Pipeline (Step by Step)

### Step 1: Preprocess the Data (Recommended)
Removes hair artifacts and resizes images — runs once and saves cleaned data to `data_processed/`:
```bash
python src/data_prepare.py
```

### Step 2: Train the Model
```bash
python src/train.py
```
- Runs for **5 epochs** by default.
- Tracks **Focal Loss** and **Accuracy** for both Train and Validation.
- Saves checkpoints to `models/`.
- Saves a `results/training_log.csv`.

### Step 3: Predict a Single Image (CLI)
```bash
python src/predict.py --image path/to/your/image.jpg
```

### Step 4: Generate a Grad-CAM Heatmap (CLI)
```bash
python src/explain.py
```
Outputs are saved to the `results/` folder.

---

## 🧠 Model Architecture

```
Input Image (224x224x3)
       │
       ▼
EfficientNet-B0 Backbone (timm)
       │
       ▼
Squeeze-and-Excitation (SE) Attention Block
  ┌──────────────────────────────┐
  │  Global Average Pool        │
  │  → FC (channels → ch/16)   │
  │  → ReLU                     │
  │  → FC (ch/16 → channels)    │
  │  → Sigmoid → Channel Scale  │
  └──────────────────────────────┘
       │
       ▼
Global Average Pooling
       │
       ▼
Dropout (0.3) → Linear (features → 10 classes)
       │
       ▼
Output (10 class probabilities)
```

### Loss Function: Focal Loss
Addresses severe class imbalance (e.g., Melanoma has 3,140 images vs. Eczema's 1,677):
```
L(p_t) = -(1 - p_t)^γ × log(p_t)
```
with `alpha=1, gamma=2`.

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **AI Framework** | PyTorch |
| **Model Backbone** | EfficientNet-B0 (via `timm`) |
| **Explainability** | `pytorch-grad-cam` |
| **Image Processing** | OpenCV, Albumentations, PIL |
| **Web Framework** | FastAPI + Uvicorn |
| **Frontend** | HTML5, CSS3 (Vanilla), JavaScript |
| **Data** | 25,331 skin disease images across 10 classes |

---

## 📊 Training Results

| Metric | Value |
|---|---|
| Epochs Trained | 5 |
| Optimizer | Adam (lr=1e-4) |
| Loss Function | Focal Loss (γ=2) |
| Train/Val Split | 80% / 20% |
| Input Image Size | 224 × 224 pixels |

> 📈 Full epoch-by-epoch logs are available at `results/training_log.csv`

---

## 🖼️ Screenshots

The dashboard features:
- 📤 **Drag & Drop** or **Live Camera** image upload
- 🔬 **Side-by-side** original vs Grad-CAM heatmap
- 📋 **Root Causes** and **Key Precautions** in plain, simple English
- ✅/❌ **Real/Fake Status**: Identifies if the uploaded image is a valid skin image

---

## ⚠️ Disclaimer

> This system is an **AI educational diagnostic aid**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified **dermatologist** or **healthcare provider** for any questions regarding a medical condition.

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

<p align="center">Made with ❤️ using PyTorch & FastAPI</p>
