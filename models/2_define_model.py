"""
Step 2: Define CNN Model Architecture
This script defines the CNN architecture for candlestick image classification
Options: Custom CNN or Transfer Learning (ResNet50)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import json

# Load dataset stats
with open('dataset/dataset_stats.json', 'r') as f:
    stats = json.load(f)

NUM_CLASSES = stats['num_classes']
IMG_SIZE = (224, 224)

print("\n" + "="*80)
print("STEP 2: DEFINING CNN ARCHITECTURE")
print("="*80 + "\n")


def create_custom_cnn(input_shape=(224, 224, 3), num_classes=3):
    """
    Custom CNN architecture for candlestick pattern recognition
    """
    print("🏗️  Building Custom CNN...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_resnet_model(input_shape=(224, 224, 3), num_classes=3, trainable_layers=10):
    """
    Transfer learning with ResNet50 (ImageNet pretrained)
    """
    print("🏗️  Building ResNet50 Transfer Learning Model...")
    
    # Load pretrained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base layers (fine-tune only last layers)
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


def get_model(model_type='custom'):
    """
    Get model based on type
    
    Args:
        model_type: 'custom' or 'resnet'
    """
    if model_type == 'custom':
        model = create_custom_cnn(num_classes=NUM_CLASSES)
        base_model = None
    elif model_type == 'resnet':
        model, base_model = create_resnet_model(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, base_model


def compile_model(model, learning_rate=0.001, class_weights=None):
    """
    Compile model with optimizer and loss
    """
    print(f"\n⚙️  Compiling model (LR: {learning_rate})...")
    
    # Calculate class weights if imbalanced
    if class_weights is None:
        # Balanced: Buy 26.5%, Sell 26.9%, Hold 46.6%
        # Weight = 1 / frequency
        total = sum(stats['distribution']['overall'].values())
        class_weights = {
            0: total / stats['distribution']['overall']['Buy'],    # Buy
            1: total / stats['distribution']['overall']['Sell'],   # Sell
            2: total / stats['distribution']['overall']['Hold']    # Hold
        }
        print(f"  📊 Class weights: {class_weights}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',  # For one-hot encoded labels
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model, class_weights


if __name__ == "__main__":
    # Test model creation
    print("\n🧪 Testing model architectures...\n")
    
    # Custom CNN
    print("1️⃣  Custom CNN:")
    model_custom, _ = get_model('custom')
    print(f"  ✅ Parameters: {model_custom.count_params():,}")
    
    # ResNet50
    print("\n2️⃣  ResNet50 Transfer Learning:")
    model_resnet, base = get_model('resnet')
    print(f"  ✅ Total parameters: {model_resnet.count_params():,}")
    print(f"  ✅ Trainable parameters: {sum([tf.size(v).numpy() for v in model_resnet.trainable_variables]):,}")
    
    # Compile
    model_custom, weights = compile_model(model_custom)
    print(f"\n✅ Model compilation successful!")
    
    # Summary
    print("\n📋 Model Summary (Custom CNN):")
    print("="*80)
    model_custom.summary()
    
    print("\n" + "="*80)
    print("✅ ARCHITECTURE DEFINITION COMPLETE!")
    print("="*80 + "\n")
