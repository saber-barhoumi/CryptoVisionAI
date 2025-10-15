# 🚀 CNN Training Pipeline - Quick Start Guide

## 📋 Overview
Complete pipeline for training a CNN model on candlestick images:
- **Dataset**: 260,000 balanced candlestick images (Buy/Sell/Hold)
- **Architecture**: Custom CNN or ResNet50 transfer learning
- **Training**: Data augmentation, callbacks, monitoring

---

## 🎯 Step-by-Step Execution

### **Step 1: Prepare Dataset** (5-10 minutes)
Creates train/val/test splits and labels.csv

```powershell
cd models
python 1_prepare_dataset.py
```

**Output:**
- `dataset/labels.csv` - Full dataset with labels
- `dataset/train.csv` - Training set (70%)
- `dataset/val.csv` - Validation set (15%)
- `dataset/test.csv` - Test set (15%)
- `dataset/dataset_stats.json` - Statistics

---

### **Step 2: Define Model** (1 minute)
Test model architectures

```powershell
python 2_define_model.py
```

**Output:**
- Model summaries printed
- Architecture validation

---

### **Step 3: Train Model** (2-8 hours depending on hardware)
Complete training with monitoring

```powershell
python 3_train_model.py
```

**Output:**
- `saved_models/{run_name}/best_model.keras` - Best model
- `saved_models/{run_name}/final_model.keras` - Final model
- `saved_models/{run_name}/training_history.csv` - Training logs
- `saved_models/{run_name}/test_results.json` - Test metrics
- `saved_models/{run_name}/training_curves.png` - Visualization
- `logs/{run_name}/` - TensorBoard logs

---

## 📊 Monitor Training

### View TensorBoard (Real-time)
```powershell
tensorboard --logdir="logs"
```
Open: http://localhost:6006

### Check Training Progress
```powershell
# View latest logs
Get-Content "saved_models\{run_name}\training_log.csv" -Tail 10

# Check test results
Get-Content "saved_models\{run_name}\test_results.json"
```

---

## ⚙️ Configuration

Edit training parameters in `3_train_model.py`:

```python
MODEL_TYPE = 'custom'  # 'custom' or 'resnet'
BATCH_SIZE = 32        # Increase for faster training (if GPU memory allows)
EPOCHS = 50            # Maximum epochs (early stopping may end sooner)
INITIAL_LR = 0.001     # Learning rate
```

---

## 🎯 Expected Results

### Custom CNN:
- **Parameters**: ~5-10M
- **Training time**: 2-4 hours (GPU), 10-20 hours (CPU)
- **Target accuracy**: 50-70%

### ResNet50 Transfer Learning:
- **Parameters**: ~25M
- **Training time**: 4-8 hours (GPU), 20-40 hours (CPU)
- **Target accuracy**: 60-80%

---

## 🔧 Requirements

### Install TensorFlow (if not installed)
```powershell
pip install tensorflow
pip install tensorboard
```

### Verify GPU (optional but recommended)
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

---

## 📁 Directory Structure After Training

```
Vision Model (CNN)/
├── models/
│   ├── 1_prepare_dataset.py
│   ├── 2_define_model.py
│   ├── 3_train_model.py
│   └── __init__.py
├── dataset/
│   ├── labels.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── dataset_stats.json
├── saved_models/
│   └── custom_cnn_20240101_120000/
│       ├── best_model.keras
│       ├── final_model.keras
│       ├── training_history.csv
│       ├── training_log.csv
│       ├── test_results.json
│       └── training_curves.png
├── logs/
│   └── custom_cnn_20240101_120000/
│       └── (TensorBoard files)
└── Candlestick_Images_Balanced/
    └── (260,000 images)
```

---

## 🚀 Quick Commands Cheatsheet

```powershell
# Full pipeline
cd models
python 1_prepare_dataset.py && python 2_define_model.py && python 3_train_model.py

# Monitor training (separate terminal)
tensorboard --logdir="logs"

# After training - commit to GitHub
cd ..
.\smart-commit.ps1

# Push automatically
.\smart-commit.ps1 -Push
```

---

## 🐛 Troubleshooting

### Issue: TensorFlow not found
**Solution:**
```powershell
pip install tensorflow
```

### Issue: Out of memory during training
**Solution:** Reduce batch size in `3_train_model.py`:
```python
BATCH_SIZE = 16  # or 8 for very limited memory
```

### Issue: Training too slow
**Solutions:**
1. Use GPU (install `tensorflow-gpu`)
2. Reduce dataset size temporarily
3. Use smaller model (Custom CNN instead of ResNet50)

### Issue: Poor accuracy (<40%)
**Possible causes:**
1. Need more epochs (increase `EPOCHS`)
2. Need different learning rate (adjust `INITIAL_LR`)
3. Need data augmentation tuning

---

## 📈 Next Steps After Training

1. **Evaluate model**: Check test_results.json
2. **Visualize**: Open TensorBoard logs
3. **Optimize**: Fine-tune hyperparameters
4. **Deploy**: Use best_model.keras for predictions
5. **Backtest**: Test on real trading scenarios

---

## 🎓 Understanding the Pipeline

### Data Augmentation (Training Only)
- Small rotations (±5°)
- Width/height shifts (±5%)
- Zoom (±5%)
- No horizontal flip (time matters!)

### Class Weights
Automatically calculated to handle slight imbalance:
- Buy: 26.5% → weight ≈ 1.13
- Sell: 26.9% → weight ≈ 1.11
- Hold: 46.6% → weight ≈ 0.64

### Callbacks
- **ModelCheckpoint**: Saves best model (highest val_accuracy)
- **EarlyStopping**: Stops if no improvement for 10 epochs
- **ReduceLROnPlateau**: Reduces learning rate if stuck
- **TensorBoard**: Real-time training visualization
- **CSVLogger**: Saves metrics to CSV

---

## 💡 Pro Tips

1. **Start with Custom CNN**: Faster to train, good baseline
2. **Use GPU**: 5-10x faster training
3. **Monitor TensorBoard**: Catch overfitting early
4. **Commit regularly**: Use smart-commit.ps1 after each step
5. **Save experiments**: Each run gets unique timestamp folder

---

## 📞 Support

Having issues? Check:
1. Python version (3.8+ required)
2. TensorFlow installed correctly
3. Dataset prepared (Step 1 completed)
4. Sufficient disk space (~2GB for models)

---

**Ready to train?** Start with Step 1! 🚀

```powershell
cd models
python 1_prepare_dataset.py
```
