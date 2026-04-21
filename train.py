import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model import SkinLesionModel, FocalLoss
from preprocess import remove_hair
import os
import torch.optim as optim
from PIL import Image
import numpy as np

import pandas as pd
import time

def train_model(data_dir='data_processed'):
    # Hyperparameters
    num_epochs = 5
    batch_size = 16
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check for processed data first
    if os.path.exists('data_processed') and os.listdir('data_processed'):
        data_dir = 'data_processed'
        print(f"Using pre-processed data from: {data_dir}")
    else:
        print(f"Using raw data from: {data_dir}. Note: Pre-processing is done on-the-fly or skipped for efficiency.")
    
    # 1. Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Dataset
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: Data directory '{data_dir}' is empty or not found.")
        return

    full_dataset = ImageFolder(root=data_dir, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes: {full_dataset.classes}")

    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. Model, Loss, Optimizer
    model = SkinLesionModel(num_classes=num_classes, pretrained=True).to(device)
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For logging
    history = []
    log_file = 'results/training_log.csv'
    if not os.path.exists('results'):
        os.makedirs('results')

    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {duration:.2f}s")
        print(f"  Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.4f}")
        
        # Log metrics
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })
        pd.DataFrame(history).to_csv(log_file, index=False)

        # Save model checkpoint
        if not os.path.exists('models'):
            os.makedirs('models')
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/skin_lesion_model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "models/skin_lesion_final.pth")
    print(f"Training complete. Log saved to {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train skin disease model.")
    parser.add_argument("--data", type=str, default="data", help="Directory containing images")
    args = parser.parse_args()
    
    train_model(args.data)
