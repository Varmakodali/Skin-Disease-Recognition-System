import pandas as pd
import matplotlib.pyplot as plt

def plot_history(csv_path='results/training_log.csv'):
    try:
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#FF4B4B', linewidth=2)
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#1C1C1C', linestyle='--')
        plt.title('Training & Validation Loss', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Acc', color='#4CAF50', linewidth=2)
        plt.plot(df['epoch'], df['val_acc'], label='Val Acc', color='#1C1C1C', linestyle='--')
        plt.title('Training & Validation Accuracy', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/model_performance.png')
        print("Success: Performance chart saved to results/model_performance.png")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_history()
