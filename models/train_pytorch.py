"""
PyTorch CNN Training Pipeline
Works with the same dataset prepared by 1_prepare_dataset.py
Alternative to TensorFlow with better Windows compatibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
BASE_DIR = Path(r"C:\Users\saber\Desktop\1trading\Vision Model (CNN)")
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = BASE_DIR / "Candlestick_Images_Balanced"
MODELS_DIR = BASE_DIR / "saved_models_pytorch"
LOGS_DIR = BASE_DIR / "logs_pytorch"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Training parameters
MODEL_TYPE = 'custom'  # 'custom' or 'resnet'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*80)
print("PYTORCH CNN TRAINING PIPELINE")
print("="*80 + "\n")
print(f"🖥️  Device: {DEVICE}")

# Load dataset stats
with open(DATASET_DIR / 'dataset_stats.json', 'r') as f:
    stats = json.load(f)

print(f"📊 Dataset: {stats['total_images']:,} images")
print(f"  - Train: {stats['train_images']:,}")
print(f"  - Val:   {stats['val_images']:,}")
print(f"  - Test:  {stats['test_images']:,}")


# Custom Dataset class
class CandlestickDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.label_mapping = {'Buy': 0, 'Sell': 1, 'Hold': 2}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.images_dir / row['image_path']
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_mapping[row['label']]
        
        return image, label


# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
print("\n🔄 Creating datasets...")
train_dataset = CandlestickDataset(
    DATASET_DIR / 'train.csv',
    IMAGES_DIR,
    transform=train_transform
)
val_dataset = CandlestickDataset(
    DATASET_DIR / 'val.csv',
    IMAGES_DIR,
    transform=val_transform
)
test_dataset = CandlestickDataset(
    DATASET_DIR / 'test.csv',
    IMAGES_DIR,
    transform=val_transform
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✅ Datasets created:")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches:   {len(val_loader)}")
print(f"  - Test batches:  {len(test_loader)}")


# Define Custom CNN
class CustomCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_resnet_model(num_classes=3):
    """Transfer learning with ResNet50"""
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


# Create model
print(f"\n🏗️  Building {MODEL_TYPE.upper()} model...")
if MODEL_TYPE == 'custom':
    model = CustomCNN(num_classes=3)
else:
    model = create_resnet_model(num_classes=3)

model = model.to(DEVICE)
print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Calculate class weights
total = sum(stats['distribution']['overall'].values())
class_weights = torch.tensor([
    total / stats['distribution']['overall']['Buy'],
    total / stats['distribution']['overall']['Sell'],
    total / stats['distribution']['overall']['Hold']
], dtype=torch.float32).to(DEVICE)

print(f"\n⚖️  Class weights: {class_weights.cpu().numpy()}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(loader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total


# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


# Training loop
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80 + "\n")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{MODEL_TYPE}_cnn_pytorch_{timestamp}"
run_dir = MODELS_DIR / run_name
run_dir.mkdir(exist_ok=True)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 40)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), run_dir / 'best_model.pth')
        print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")

# Test evaluation
print("\n" + "="*80)
print("📊 EVALUATING ON TEST SET")
print("="*80 + "\n")

model.load_state_dict(torch.load(run_dir / 'best_model.pth'))
test_loss, test_acc = validate(model, test_loader, criterion)

print(f"\n✅ Test Results:")
print(f"  - Loss:     {test_loss:.4f}")
print(f"  - Accuracy: {test_acc:.2f}%")

# Save results
results = {
    'model_type': MODEL_TYPE,
    'framework': 'pytorch',
    'timestamp': timestamp,
    'device': str(DEVICE),
    'epochs_completed': EPOCHS,
    'best_val_accuracy': best_val_acc,
    'test_accuracy': test_acc,
    'test_loss': test_loss
}

with open(run_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save history
pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Validation')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'], label='Validation')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(run_dir / 'training_curves.png', dpi=150)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📁 Output: {run_dir}")
print(f"\n🎯 Results:")
print(f"  - Best Val Accuracy: {best_val_acc:.2f}%")
print(f"  - Test Accuracy:     {test_acc:.2f}%")
print("="*80 + "\n")
