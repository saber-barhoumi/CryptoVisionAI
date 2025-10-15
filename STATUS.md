# 🎯 CNN Training Pipeline - COMPLETE SETUP

## ✅ What's Ready

### 1. **Dataset Prepared** (254,424 images)
- ✅ Labels.csv created
- ✅ Train/Val/Test split (70/15/15)
- ✅ Balanced distribution (26.5% Buy, 26.9% Sell, 46.6% Hold)
- ✅ All images organized by pair/label

### 2. **Training Scripts Ready**
- ✅ `models/train_pytorch.py` - PyTorch CNN (recommended)
- ✅ `models/3_train_model.py` - TensorFlow CNN
- ✅ `models/1_prepare_dataset.py` - Dataset preparation (completed)
- ✅ `models/2_define_model.py` - Model architectures

### 3. **Installation Scripts**
- ✅ `setup-training.ps1` - Interactive framework selection
- ✅ `install-pytorch.ps1` - Install PyTorch (recommended)
- ✅ `fix-tensorflow.ps1` - Fix TensorFlow installation issues

### 4. **Documentation**
- ✅ `QUICKSTART.md` - Fast track guide
- ✅ `TRAINING_GUIDE.md` - Complete training guide
- ✅ `requirements_training.txt` - Dependencies list

### 5. **GitHub**
- ✅ All files committed
- ✅ Pushed to: https://github.com/saber-barhoumi/CryptoVisionAI
- ✅ Professional commit history

---

## 🚀 Next Step: START TRAINING!

### Option A: PyTorch (RECOMMENDED for Windows)

```powershell
# Step 1: Install PyTorch
.\install-pytorch.ps1

# Step 2: Train
cd models
python train_pytorch.py
```

**Advantages:**
- ✅ Better Windows compatibility
- ✅ Faster installation (no wheel.exe issues)
- ✅ Same accuracy as TensorFlow
- ✅ Modern and flexible

---

### Option B: TensorFlow (If PyTorch fails)

```powershell
# Step 1: Fix installation
.\fix-tensorflow.ps1

# Step 2: Train
cd models
python 3_train_model.py
```

---

## 📊 Training Details

### What happens during training:
1. **Data Loading**: 254K images loaded in batches
2. **Augmentation**: Random rotation, shifts (training only)
3. **Training**: Forward pass → loss → backprop → update
4. **Validation**: Check accuracy on unseen data
5. **Early stopping**: Stops if no improvement
6. **Checkpointing**: Saves best model automatically

### Expected output:
- **Epoch 1/50**: Train loss: 1.0234, Val acc: 45.23%
- **Epoch 10/50**: Train loss: 0.7891, Val acc: 58.67%
- **Epoch 30/50**: Train loss: 0.5234, Val acc: 65.12% ✅ Best
- **Final**: Test acc: 64.87%

### Files generated:
```
saved_models_pytorch/
└── custom_cnn_pytorch_20241015_143022/
    ├── best_model.pth         # Best model (use this!)
    ├── training_curves.png    # Accuracy/loss graphs
    ├── history.csv            # Training logs
    └── results.json           # Final metrics
```

---

## ⚙️ Customization

### Change batch size (if out of memory):
```python
# In train_pytorch.py or 3_train_model.py
BATCH_SIZE = 16  # Default: 32, Try: 16, 8
```

### Change epochs:
```python
EPOCHS = 20  # Default: 50, For testing: 10-20
```

### Change model:
```python
MODEL_TYPE = 'resnet'  # Default: 'custom', Options: 'custom', 'resnet'
```

---

## 📈 Monitoring Training

### Real-time progress:
```
Epoch 1/50
Training: 100%|████████| 5566/5566 [12:34<00:00, loss: 0.8234, acc: 52.34%]
Validation: 100%|████████| 1193/1193 [02:45<00:00]

Train Loss: 0.8234 | Train Acc: 52.34%
Val Loss:   0.7891 | Val Acc:   55.67%
✅ Saved best model (Val Acc: 55.67%)
```

### View curves:
Open `training_curves.png` after training completes

---

## 🎓 Understanding Results

### Good signs:
- ✅ Val accuracy increasing
- ✅ Train/Val loss decreasing
- ✅ No huge gap between train and val accuracy (<10%)

### Warning signs:
- ⚠️ Val accuracy stuck/decreasing → Overfitting
- ⚠️ Train accuracy 90%, Val 40% → Too much overfitting
- ⚠️ Both accuracies low (<40%) → Need more epochs or different model

### Solutions:
1. **Overfitting**: Add more dropout, augmentation
2. **Underfitting**: More epochs, bigger model, lower learning rate
3. **Stuck**: Reduce learning rate, change optimizer

---

## 🐛 Common Issues

### Issue: ModuleNotFoundError: No module named 'tensorflow'
**Solution**: Use PyTorch instead:
```powershell
.\install-pytorch.ps1
cd models
python train_pytorch.py
```

### Issue: Out of memory
**Solution**: Reduce batch size:
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Training too slow
**Solutions**:
1. Reduce epochs: `EPOCHS = 20`
2. Use GPU (if available)
3. Reduce dataset temporarily

### Issue: Low accuracy (<40%)
**Solutions**:
1. Increase epochs: `EPOCHS = 100`
2. Try different learning rate: `LEARNING_RATE = 0.0001`
3. Use ResNet: `MODEL_TYPE = 'resnet'`

---

## 💡 Pro Tips

1. **Start small**: Train 10 epochs first to test everything works
2. **Monitor**: Watch training curves - they tell you everything
3. **Patience**: CNN training takes time (2-6 hours normal)
4. **GPU**: If you have NVIDIA GPU, install CUDA version
5. **Backup**: Models are saved automatically in `saved_models_pytorch/`

---

## 🎯 After Training

### 1. Evaluate results:
```python
# Check test accuracy
cat saved_models_pytorch/*/results.json
```

### 2. Commit to GitHub:
```powershell
git add .
git commit -m "✅ Trained CNN model - Test accuracy: 65.2%"
git push
```

### 3. Use model for predictions:
```python
import torch
model = CustomCNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
# Now predict on new images!
```

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Install PyTorch | `.\install-pytorch.ps1` |
| Install TensorFlow | `.\fix-tensorflow.ps1` |
| Train (PyTorch) | `cd models ; python train_pytorch.py` |
| Train (TensorFlow) | `cd models ; python 3_train_model.py` |
| Check dataset | `cat dataset\dataset_stats.json` |
| Commit changes | `git add . ; git commit -m "message" ; git push` |

---

## 🌟 Summary

You now have:
- ✅ **254,424 balanced images** ready for training
- ✅ **70/15/15 split** (train/val/test)
- ✅ **2 training options** (PyTorch + TensorFlow)
- ✅ **Complete pipeline** (data → train → evaluate → save)
- ✅ **Professional GitHub repo** with all files
- ✅ **Clear documentation** for every step

**Everything is ready. Just choose your framework and start training!** 🚀

---

**Recommended first step:**

```powershell
.\install-pytorch.ps1
```

Then after installation completes:

```powershell
cd models
python train_pytorch.py
```

**Let it run for 2-6 hours, and you'll have a trained CNN model!** 🎉

---

📝 **Last updated**: October 15, 2025  
🔗 **GitHub**: https://github.com/saber-barhoumi/CryptoVisionAI  
👨‍💻 **Author**: Saber Barhoumi
