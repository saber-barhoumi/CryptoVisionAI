# 🚀 GPU Training - Quick Start (RTX 4050)

## 🎮 Your GPU: NVIDIA GeForce RTX 4050 Laptop GPU

**Great news!** Your GPU is detected and ready to use.

---

## ⚡ Expected Training Time

| Model Type | CPU Time | GPU Time (RTX 4050) |
|------------|----------|---------------------|
| Custom CNN | 10-20h | **2-4 hours** ⚡ |
| ResNet50 | 20-40h | **4-6 hours** ⚡ |

**Your GPU is 5-10x faster than CPU!**

---

## 🚀 Start Training NOW (GPU-Optimized)

### Method 1: Interactive (Recommended)
```powershell
.\START.ps1
```
Select option **[1] Start Training**

### Method 2: Direct
```powershell
# Install PyTorch (if not already installed)
.\install-pytorch.ps1

# Start training (GPU will be used automatically)
cd models
python train_pytorch.py
```

---

## 🎯 GPU Training Tips

### 1. **Maximize GPU Usage**
Edit `train_pytorch.py` and increase batch size:
```python
BATCH_SIZE = 64  # Default: 32, Try: 64, 128
```
**Larger batch = faster training!**

### 2. **Monitor GPU Usage**
Open a new PowerShell window:
```powershell
# Check GPU utilization (install if needed: pip install gpustat)
nvidia-smi
```

### 3. **Keep Laptop Cool**
- Use cooling pad (important for laptop GPUs)
- Don't run other GPU-intensive apps
- Ensure good ventilation

### 4. **Optimal Settings for RTX 4050**
```python
# In train_pytorch.py
BATCH_SIZE = 64    # RTX 4050 can handle this
EPOCHS = 50        # Full training
NUM_WORKERS = 4    # Faster data loading
```

---

## 📊 What to Expect

### During Training:
```
Epoch 1/50
Training: 100%|████████| 2783/2783 [06:15<00:00, 7.41it/s, loss: 0.8234, acc: 52.34%]
Validation: 100%|████████| 597/597 [01:20<00:00, 7.46it/s]

Train Loss: 0.8234 | Train Acc: 52.34%
Val Loss:   0.7891 | Val Acc:   55.67%
✅ Saved best model (Val Acc: 55.67%)
```

**Each epoch: ~7-8 minutes** (vs 30-40 min on CPU)

---

## 🔧 GPU Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"
**Solution**: Reduce batch size
```python
BATCH_SIZE = 32  # or 16
```

### Issue: GPU not being used
**Check**:
```python
python -c "import torch; print(torch.cuda.is_available())"
```
If False, reinstall PyTorch with CUDA support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Training slower than expected
**Solutions**:
1. Close other applications (Chrome, games, etc.)
2. Update NVIDIA drivers
3. Increase batch size to 64 or 128

---

## 💡 Advanced: Monitor Training

### Real-time GPU stats:
```powershell
# In a separate PowerShell window
while ($true) { 
    Clear-Host
    nvidia-smi
    Start-Sleep -Seconds 2
}
```

### Expected GPU usage:
- **Utilization**: 80-95%
- **Memory**: 3-4 GB (out of 6 GB)
- **Temperature**: 65-80°C

---

## 🎯 Recommended Training Command

```powershell
# Full GPU-optimized training
cd models
python train_pytorch.py

# Or with custom settings:
# Edit train_pytorch.py first:
# BATCH_SIZE = 64
# EPOCHS = 50
```

---

## ⏱️ Estimated Timeline

| Stage | Time |
|-------|------|
| Dataset loading | 1 min |
| Epoch 1-10 | 70 min |
| Epoch 11-30 | 140 min |
| Epoch 31-50 | 140 min |
| **Total** | **~6 hours** |

With early stopping, likely **3-4 hours** total.

---

## 🎉 Why This is Great

✅ **RTX 4050** = Perfect for deep learning  
✅ **6GB VRAM** = Enough for this dataset  
✅ **254K images** = Will train efficiently  
✅ **PyTorch** = Already configured for GPU  

**You're all set! Just run `.\START.ps1` and choose option 1!** 🚀

---

## 📊 After Training

Your trained model will be saved in:
```
saved_models_pytorch/
└── custom_cnn_pytorch_YYYYMMDD_HHMMSS/
    ├── best_model.pth          # Use this for predictions!
    ├── training_curves.png     # View your training progress
    ├── history.csv             # Detailed metrics
    └── results.json            # Final accuracy
```

**Expected accuracy: 55-70%** (for 3-class trading problem)

---

**Ready to start?**

```powershell
.\START.ps1
```

Choose **[1] Start Training** and let your RTX 4050 do the work! ⚡
