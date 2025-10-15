# 🚀 Quick Start: Train Your CNN Model# 🚀 QUICK START GUIDE - Process ALL Your Binance Data



## ⚡ Fast Track (Recommended)## 📊 What You Have



### Option 1: PyTorch (Best for Windows) ✅✅ **1,000 parquet files** with 1.5 BILLION candles of trading data

✅ **272 USDT pairs** (most liquid and best for training)

```powershell✅ **All scripts ready** to process everything

# 1. Install PyTorch

.\install-pytorch.ps1---



# 2. Train model (uses existing dataset)## 🎯 Choose Your Strategy

cd models

python train_pytorch.py### Option 1: RECOMMENDED - USDT Pairs (15-20 hours)

```**Best for first serious training**

- 272 files (USDT pairs only)

**Why PyTorch?**- ~136,000 images

- ✅ Better Windows compatibility- ~250 MB storage

- ✅ Faster installation- Better balanced labels

- ✅ Same results as TensorFlow

- ✅ Works with your 254K images dataset### Option 2: Major Pairs (30 hours)

**More variety**

---- 517 files (USDT + BTC)

- ~258,500 images

### Option 2: TensorFlow/Keras- ~500 MB storage



```powershell### Option 3: Everything (2.4 days)

# 1. Fix installation issues**Maximum dataset**

.\fix-tensorflow.ps1- All 1,000 files

- ~1,000,000 images

# 2. Train model- ~1.7 GB storage

cd models

python 3_train_model.py---

```

## ⚡ STEP-BY-STEP INSTRUCTIONS

---

### Step 1: Choose Strategy

## 📊 What's Already Done

Open `config_full_processing.py` and uncomment the strategy you want:

✅ **Dataset prepared** (254,424 images)

- Train: 178,096 (70%)**For USDT pairs (RECOMMENDED):**

- Val: 38,164 (15%)```python

- Test: 38,164 (15%)# Find these lines and uncomment them (remove the # at the start):

WINDOW_SIZE = 30

✅ **Balanced distribution**FUTURE_BARS = 5

- Buy: 26.5%THRESHOLD_PERCENT = 0.3  # Better balance

- Sell: 26.9%IMAGE_WIDTH = 224

- Hold: 46.6%IMAGE_HEIGHT = 224

DPI = 100

✅ **Files ready**MAX_FILES = None  # Process all USDT files

- `dataset/train.csv` - Training dataMAX_IMAGES_PER_FILE = 500

- `dataset/val.csv` - Validation dataQUOTE_CURRENCIES = ['USDT']  # Only USDT pairs

- `dataset/test.csv` - Test dataDATA_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Binance Full History 28gb"

OUTPUT_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_USDT"

---```



## 🎯 Training Output**For ALL files:**

```python

After training, you'll get:# Uncomment STRATEGY 3 instead

- `best_model.pth` (PyTorch) or `best_model.keras` (TensorFlow)WINDOW_SIZE = 30

- `training_curves.png` - VisualizationFUTURE_BARS = 5

- `results.json` - MetricsTHRESHOLD_PERCENT = 0.3

- `history.csv` - Training logsIMAGE_WIDTH = 224

IMAGE_HEIGHT = 224

---DPI = 100

MAX_FILES = None  # All files

## 💡 Quick CommandsMAX_IMAGES_PER_FILE = 1000

QUOTE_CURRENCIES = []  # All currencies

```powershellDATA_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Binance Full History 28gb"

# Choose framework interactivelyOUTPUT_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Full"

.\setup-training.ps1```



# Or install directly### Step 2: Run the Processor

.\install-pytorch.ps1      # Recommended

.\fix-tensorflow.ps1       # If you prefer TensorFlowOpen PowerShell and run:



# Then train```powershell

cd modelscd "c:\Users\saber\Desktop\1trading\Vision Model (CNN)"

python train_pytorch.py    # PyTorchpython process_full_dataset.py

# OR```

python 3_train_model.py    # TensorFlow

```### Step 3: Let It Run



---The script will:

- ✅ Show progress bar

## 🐛 Troubleshooting- ✅ Save checkpoints every 50 files (can resume if interrupted)

- ✅ Display statistics periodically

**TensorFlow won't install?**- ✅ Create organized folder structure

→ Use PyTorch instead: `.\install-pytorch.ps1`

**You can:**

**Out of memory during training?**- Let it run overnight

→ Edit `BATCH_SIZE = 16` (or 8) in training script- Close terminal and resume later (checkpoints saved)

- Monitor progress in real-time

**Training too slow?**

→ Reduce `EPOCHS = 20` for testing---



---## 📁 Output Structure



## 📈 Expected Results```

Candlestick_Images_USDT/

- **Training time**: 2-6 hours (depends on CPU/GPU)├── Buy/

- **Expected accuracy**: 50-70% (3-class problem)│   ├── BTC-USDT/

- **GPU**: 5-10x faster than CPU│   │   ├── Buy_BTC-USDT_00000100.png

│   │   ├── Buy_BTC-USDT_00000250.png

---│   │   └── ... (thousands more)

│   ├── ETH-USDT/

## 🚀 After Training│   ├── BNB-USDT/

│   └── ... (272 pairs)

1. **Commit to GitHub**:├── Sell/

   ```powershell│   └── ... (same structure)

   .\smart-commit.ps1├── Hold/

   ```│   └── ... (same structure)

└── checkpoint.json (for resume)

2. **View results**:```

   - Check `saved_models_pytorch/` or `saved_models/`

   - Open `training_curves.png`---



3. **Use model for predictions**:## 🛠️ Resume If Interrupted

   - Load `best_model.pth` (PyTorch)

   - Or `best_model.keras` (TensorFlow)If the process stops for any reason:



---```powershell

# Just run it again - it will resume from checkpoint!

**Ready? Start with PyTorch!** 🎉python process_full_dataset.py

```

```powershell

.\install-pytorch.ps1The script automatically:

```- Loads checkpoint.json

- Skips already processed files
- Continues where it left off

---

## 📊 Monitor Progress

The script shows:
```
Processing: 45%|████████▌        | 123/272 [01:24<01:49, 2.23it/s]

Progress Update:
  Files: 123
  Images: 62,150 (Buy: 28.3% | Sell: 31.2% | Hold: 40.5%)
```

**Much better balance with threshold=0.3%!**

---

## ⏱️ Time Estimates

| Strategy | Files | Time | When Done? |
|----------|-------|------|------------|
| USDT (500/file) | 272 | 15-20h | Tomorrow morning |
| Major (500/file) | 517 | 30h | Day after tomorrow |
| All (1000/file) | 1,000 | 58h | 2-3 days |

---

## 💾 Check Disk Space

Before starting, make sure you have enough space:

```powershell
# Check free space on C: drive
Get-PSDrive C
```

You need:
- USDT: ~300 MB free
- Major: ~600 MB free
- All: ~2 GB free

---

## 🎯 After Processing

Once complete, you'll have:

### For USDT Strategy (~136,000 images):
```
Buy:   ~38,000 images (28%)
Sell:  ~40,000 images (29%)
Hold:  ~58,000 images (43%)
```

### Split for Training:
```
Training:   95,200 images (70%)
Validation: 20,400 images (15%)
Test:       20,400 images (15%)
```

Perfect for CNN training! 🎉

---

## 🚀 START NOW!

### Quick Commands:

```powershell
# 1. Go to directory
cd "c:\Users\saber\Desktop\1trading\Vision Model (CNN)"

# 2. Edit config_full_processing.py (uncomment your strategy)
notepad config_full_processing.py

# 3. Run processor
python process_full_dataset.py

# 4. Go do something else - let it run!
```

---

## 🔍 Test First (RECOMMENDED)

Before processing everything, test with 20 files:

1. Open `config_full_processing.py`
2. Scroll to "TEST MODE"
3. Uncomment those lines instead
4. Run: `python process_full_dataset.py`
5. Check output in ~10 minutes
6. If looks good, switch to full strategy

---

## ❓ FAQ

**Q: Can I stop and resume?**
A: Yes! Checkpoints saved every 50 files.

**Q: How do I check progress?**
A: Look at the progress bar in terminal.

**Q: What if I run out of space?**
A: Reduce `MAX_IMAGES_PER_FILE` in config.

**Q: Can I process while doing other work?**
A: Yes, but it will use CPU. Consider running overnight.

**Q: Is threshold 0.3% better than 0.5%?**
A: Yes! 0.3% gives ~30/30/40 distribution vs 5/8/87.

---

## 🎓 Next Steps After Processing

1. **Verify images** - Check a few to make sure they look good
2. **Split dataset** - Train/Val/Test (70/15/15)
3. **Build CNN** - Use TensorFlow or PyTorch
4. **Train model** - Start with ResNet or VGG architecture
5. **Backtest** - Test on historical data before live trading

---

## 📞 Need Help?

If you encounter issues:
1. Check error messages in terminal
2. Look at checkpoint.json for progress
3. Try TEST MODE first
4. Reduce MAX_IMAGES_PER_FILE if too slow

---

**Ready? Let's process that data! 🚀📈**

Run: `python process_full_dataset.py`
