import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


class ChestXRayDataset(Dataset):
    """
    Dataset for Chest X-Ray images.
    
    Loads images from train folder with NORMAL and PNEUMONIA subfolders.
    Applies transforms for preprocessing.
    """
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.images = []
        self.labels = []
        
        # Load images from subfolders
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(image_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def train_cnn():
    """
    Train chest X-ray classification model using Kaggle dataset.
    
    Pipeline:
    1. Load data from chest_xray/chest_xray/train/
    2. Data verification (image counts)
    3. Preprocessing (resize, normalize)
    4. Create 80/20 train-val split
    5. Train ResNet18 model
    6. Evaluate with multiple metrics
    7. Save model
    """
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # =============================================================================
    # Step 1: Load Data
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    # Data path - using train folder from Kaggle dataset
    data_dir = 'data/chest_xray/chest_xray/train'
    print(f"Data loaded from: {data_dir}")
    
    # =============================================================================
    # Step 2: Data Verification & Preprocessing
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Data Verification & Preprocessing")
    print("=" * 60)
    
    # Define transforms for preprocessing
    # Resize images to 224x224 and normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = ChestXRayDataset(data_dir, transform=transform)
    
    print(f"Dataset loaded successfully [OK]")
    print(f"Total images: {len(dataset)}")
    
    # Count per class
    normal_count = dataset.labels.count(0)
    pneumonia_count = dataset.labels.count(1)
    print(f"  - NORMAL: {normal_count}")
    print(f"  - PNEUMONIA: {pneumonia_count}")
    
    # =============================================================================
    # Step 3: Train-Validation Split (80/20)
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Train-Validation Split (80/20)")
    print("=" * 60)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print("Data loaders created [OK]")
    
    # =============================================================================
    # Step 4: Model Setup
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Model Setup (ResNet18)")
    print("=" * 60)
    
    # Load ResNet18 pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    
    # Modify final layer for binary classification
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    print("Model: ResNet18 (pretrained on ImageNet)")
    print("Modified final layer for 2 classes: NORMAL, PNEUMONIA")
    print("Optimizer: Adam (lr=0.001)")
    print("Scheduler: StepLR (step_size=3, gamma=0.1)")
    
    # =============================================================================
    # Step 5: Training
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training (10 Epochs)")
    print("=" * 60)
    
    print("Starting training...\n")
    best_val_acc = 0.0
    
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/10 - Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/cnn_model.pth')
            print(f"[OK] Best model saved (val acc: {best_val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    # =============================================================================
    # Step 6: Final Evaluation
    # =============================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Final Evaluation")
    print("=" * 60)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
    model.eval()
    
    # Get predictions for validation set
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\nMetrics on Validation Set:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Pneumonia")
    print(f"Actual Normal    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Pneumonia {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: models/cnn_model.pth")


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_cnn()