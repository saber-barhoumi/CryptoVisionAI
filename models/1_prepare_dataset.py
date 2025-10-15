"""
Step 1: Prepare Dataset - Create labels.csv and split data
This script:
1. Scans all images in Candlestick_Images_Balanced
2. Creates labels.csv with image paths and labels
3. Splits data into train/val/test (70/15/15)
4. Generates dataset statistics
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = Path(r"C:\Users\saber\Desktop\1trading\Vision Model (CNN)")
IMAGES_DIR = BASE_DIR / "Candlestick_Images_Balanced"
OUTPUT_DIR = BASE_DIR / "dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

# Train/Val/Test split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

print("\n" + "="*80)
print("STEP 1: PREPARING CNN DATASET")
print("="*80 + "\n")

# Step 1: Collect all images and labels
print("📊 Scanning images...")
data = []

# The structure is: pair_name/Buy|Sell|Hold/*.png
for pair_dir in tqdm(list(IMAGES_DIR.iterdir()), desc="Scanning pairs"):
    if not pair_dir.is_dir():
        continue
    
    # Skip the top-level Buy/Sell/Hold folders (old structure)
    if pair_dir.name in ['Buy', 'Sell', 'Hold']:
        continue
    
    # Scan each label folder
    for label in ['Buy', 'Sell', 'Hold']:
        label_dir = pair_dir / label
        if not label_dir.exists():
            continue
        
        # Get all PNG files
        for img_file in label_dir.glob("*.png"):
            # Store relative path from IMAGES_DIR
            relative_path = img_file.relative_to(IMAGES_DIR)
            data.append({
                'image_path': str(relative_path).replace('\\', '/'),
                'full_path': str(img_file),
                'label': label,
                'pair': pair_dir.name
            })

print(f"\n✅ Found {len(data):,} images")

# Create DataFrame
df = pd.DataFrame(data)

# Encode labels
label_mapping = {'Buy': 0, 'Sell': 1, 'Hold': 2}
df['label_encoded'] = df['label'].map(label_mapping)

# Show distribution
print("\n📈 Label Distribution:")
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {label:5s}: {count:6,} images ({percentage:5.2f}%)")

# Step 2: Split dataset
print("\n🔀 Splitting dataset...")

# First split: train + val vs test
train_val_df, test_df = train_test_split(
    df, 
    test_size=TEST_SPLIT,
    stratify=df['label_encoded'],
    random_state=42
)

# Second split: train vs val
train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SPLIT/(TRAIN_SPLIT + VAL_SPLIT),  # Adjust for remaining data
    stratify=train_val_df['label_encoded'],
    random_state=42
)

print(f"\n✅ Split completed:")
print(f"  Train: {len(train_df):6,} images ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df):6,} images ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df):6,} images ({len(test_df)/len(df)*100:.1f}%)")

# Step 3: Save datasets
print("\n💾 Saving dataset files...")

# Save full dataset
df.to_csv(OUTPUT_DIR / 'labels.csv', index=False)
print(f"  ✅ Saved: labels.csv ({len(df):,} images)")

# Save splits
train_df.to_csv(OUTPUT_DIR / 'train.csv', index=False)
val_df.to_csv(OUTPUT_DIR / 'val.csv', index=False)
test_df.to_csv(OUTPUT_DIR / 'test.csv', index=False)
print(f"  ✅ Saved: train.csv, val.csv, test.csv")

# Step 4: Generate statistics
print("\n📊 Generating statistics...")

stats = {
    'total_images': len(df),
    'train_images': len(train_df),
    'val_images': len(val_df),
    'test_images': len(test_df),
    'num_classes': 3,
    'classes': ['Buy', 'Sell', 'Hold'],
    'label_mapping': label_mapping,
    'distribution': {
        'overall': df['label'].value_counts().to_dict(),
        'train': train_df['label'].value_counts().to_dict(),
        'val': val_df['label'].value_counts().to_dict(),
        'test': test_df['label'].value_counts().to_dict()
    },
    'num_pairs': df['pair'].nunique(),
    'split_ratio': {
        'train': TRAIN_SPLIT,
        'val': VAL_SPLIT,
        'test': TEST_SPLIT
    }
}

# Save statistics
with open(OUTPUT_DIR / 'dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"  ✅ Saved: dataset_stats.json")

# Print final summary
print("\n" + "="*80)
print("✅ DATASET PREPARATION COMPLETE!")
print("="*80)
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print(f"\n📊 Files created:")
print(f"  - labels.csv         : All {len(df):,} images with labels")
print(f"  - train.csv          : {len(train_df):,} training images")
print(f"  - val.csv            : {len(val_df):,} validation images")
print(f"  - test.csv           : {len(test_df):,} test images")
print(f"  - dataset_stats.json : Dataset statistics")
print(f"\n🎯 Ready for CNN training!")
print("="*80 + "\n")
