"""
Step 3: Train CNN Model
Complete training pipeline with data augmentation, callbacks, and monitoring
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parent))
from define_model import get_model, compile_model

# Configuration
BASE_DIR = Path(r"C:\Users\saber\Desktop\1trading\Vision Model (CNN)")
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = BASE_DIR / "Candlestick_Images_Balanced"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Training parameters
MODEL_TYPE = 'custom'  # 'custom' or 'resnet'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 0.001

print("\n" + "="*80)
print("STEP 3: TRAINING CNN MODEL")
print("="*80 + "\n")

# Load dataset stats
with open(DATASET_DIR / 'dataset_stats.json', 'r') as f:
    stats = json.load(f)

print(f"📊 Dataset: {stats['total_images']:,} images")
print(f"  - Train: {stats['train_images']:,}")
print(f"  - Val:   {stats['val_images']:,}")
print(f"  - Test:  {stats['test_images']:,}")

# Step 1: Create data generators
print("\n🔄 Creating data generators...")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,          # Small rotation (candlesticks shouldn't rotate much)
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=False,      # Don't flip (time direction matters)
    fill_mode='nearest'
)

# No augmentation for validation/test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataframes
train_df = pd.read_csv(DATASET_DIR / 'train.csv')
val_df = pd.read_csv(DATASET_DIR / 'val.csv')
test_df = pd.read_csv(DATASET_DIR / 'test.csv')

# Add full paths
train_df['full_path'] = train_df['image_path'].apply(lambda x: str(IMAGES_DIR / x))
val_df['full_path'] = val_df['image_path'].apply(lambda x: str(IMAGES_DIR / x))
test_df['full_path'] = test_df['image_path'].apply(lambda x: str(IMAGES_DIR / x))

# Create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='full_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='full_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='full_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Generators created:")
print(f"  - Train batches: {len(train_generator)}")
print(f"  - Val batches:   {len(val_generator)}")
print(f"  - Test batches:  {len(test_generator)}")

# Step 2: Build and compile model
print(f"\n🏗️  Building {MODEL_TYPE.upper()} model...")
model, base_model = get_model(MODEL_TYPE)
model, class_weights = compile_model(model, learning_rate=INITIAL_LR)

print(f"\n📊 Model parameters: {model.count_params():,}")

# Step 3: Setup callbacks
print("\n⚙️  Setting up callbacks...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{MODEL_TYPE}_cnn_{timestamp}"
run_dir = MODELS_DIR / run_name

run_dir.mkdir(exist_ok=True)

callbacks = [
    # Model checkpoint - save best model
    ModelCheckpoint(
        filepath=str(run_dir / 'best_model.keras'),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    
    # Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    
    # TensorBoard logging
    TensorBoard(
        log_dir=str(LOGS_DIR / run_name),
        histogram_freq=1,
        write_graph=True,
        write_images=False
    ),
    
    # CSV logging
    CSVLogger(
        filename=str(run_dir / 'training_log.csv'),
        separator=',',
        append=False
    )
]

print(f"  ✅ Callbacks configured")
print(f"  📁 Model save path: {run_dir}")
print(f"  📁 TensorBoard logs: {LOGS_DIR / run_name}")

# Step 4: Train model
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80 + "\n")

print(f"⏱️  Epochs: {EPOCHS}")
print(f"📦 Batch size: {BATCH_SIZE}")
print(f"🎯 Initial LR: {INITIAL_LR}")
print(f"⚖️  Class weights: {class_weights}\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Step 5: Evaluate on test set
print("\n" + "="*80)
print("📊 EVALUATING ON TEST SET")
print("="*80 + "\n")

test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    test_generator,
    verbose=1
)

# Calculate F1 score
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

print(f"\n✅ Test Results:")
print(f"  - Loss:      {test_loss:.4f}")
print(f"  - Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall:    {test_recall:.4f}")
print(f"  - F1 Score:  {test_f1:.4f}")
print(f"  - AUC:       {test_auc:.4f}")

# Step 6: Save final model and results
print("\n💾 Saving final model and results...")

# Save final model
model.save(run_dir / 'final_model.keras')
print(f"  ✅ Saved: final_model.keras")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(run_dir / 'training_history.csv', index=False)
print(f"  ✅ Saved: training_history.csv")

# Save test results
test_results = {
    'model_type': MODEL_TYPE,
    'timestamp': timestamp,
    'training': {
        'epochs_completed': len(history.history['loss']),
        'batch_size': BATCH_SIZE,
        'initial_lr': INITIAL_LR,
        'class_weights': class_weights
    },
    'test_metrics': {
        'loss': float(test_loss),
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'auc': float(test_auc)
    },
    'best_val_accuracy': float(max(history.history['val_accuracy'])),
    'best_val_loss': float(min(history.history['val_loss']))
}

with open(run_dir / 'test_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print(f"  ✅ Saved: test_results.json")

# Step 7: Plot training curves
print("\n📊 Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Validation')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Validation')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(run_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
print(f"  ✅ Saved: training_curves.png")

# Final summary
print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\n📁 Output directory: {run_dir}")
print(f"\n📊 Best Results:")
print(f"  - Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"  - Test Accuracy:     {test_accuracy:.4f}")
print(f"  - Test F1 Score:     {test_f1:.4f}")
print(f"\n🎯 Files saved:")
print(f"  - best_model.keras       : Best model (highest val accuracy)")
print(f"  - final_model.keras      : Final model after all epochs")
print(f"  - training_history.csv   : Full training history")
print(f"  - training_log.csv       : Training log per epoch")
print(f"  - test_results.json      : Test metrics and metadata")
print(f"  - training_curves.png    : Visualization of training")
print(f"\n💡 View training in TensorBoard:")
print(f"   tensorboard --logdir=\"{LOGS_DIR}\"")
print("="*80 + "\n")
