# 📚 Quick Start Guide

## Prerequisites

### 1. System Requirements

- **OS**: Windows 10+, Linux, or macOS
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 40GB free space (32GB data + 8GB processing)
- **GPU**: Optional but recommended (CUDA-compatible)

### 2. Software Requirements

```bash
Python 3.8+
pip
Git
Kaggle CLI (optional)
```

---

## Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/saber-barhoumi/CryptoVisionAI.git
cd CryptoVisionAI
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### Option A: Using Kaggle CLI (Recommended)

1. **Install Kaggle CLI:**
```bash
pip install kaggle
```

2. **Setup Kaggle API:**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Click "Create New API Token"
   - Save `kaggle.json` to:
     - Windows: `C:\Users\<YourUser>\.kaggle\kaggle.json`
     - Linux/Mac: `~/.kaggle/kaggle.json`

3. **Download Dataset:**
```bash
kaggle datasets download -d jorijnsmit/binance-full-history
```

4. **Extract:**
```bash
# Windows
Expand-Archive binance-full-history.zip -DestinationPath "Binance Full History 28gb"

# Linux/Mac
unzip binance-full-history.zip -d "Binance Full History 28gb"
```

### Option B: Manual Download

1. Go to [Binance Full History Dataset](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)
2. Click "Download" (requires Kaggle account)
3. Extract to `Binance Full History 28gb/` folder

---

## Generate Candlestick Images

### Step 1: Explore Data (Optional)

```bash
python data_preparation/explore_data.py
```

This will show:
- Number of files
- Total candles
- Date ranges
- Trading pairs

### Step 2: Calculate Statistics

```bash
python data_preparation/calculate_stats.py
```

Expected output:
```
Total files: 1,000
Total rows: 1.5 billion
USDT pairs: 272
BTC pairs: 245
```

### Step 3: Generate Images

```bash
python data_preparation/reprocess_balanced.py
```

**Processing time:** ~13 hours for full dataset

**Output:**
- Folder: `Candlestick_Images_Balanced/`
- Images: ~260,000
- Structure:
  ```
  Candlestick_Images_Balanced/
  ├── 1INCH-BTC/
  │   ├── Buy/
  │   ├── Sell/
  │   └── Hold/
  ├── AAVE-USDT/
  │   ├── Buy/
  │   ├── Sell/
  │   └── Hold/
  ...
  ```

### Step 4: Verify Balance

**Windows:**
```powershell
.\scripts\check_balance.ps1
```

**Linux/Mac:**
```bash
python scripts/check_balance.py
```

Expected distribution:
```
Buy:  26-30%
Sell: 26-30%
Hold: 40-48%
```

---

## Train CNN Model

### Step 1: Configure Training

Edit `config.py`:

```python
# Model settings
MODEL_TYPE = 'resnet50'  # or 'custom', 'vgg16', 'efficientnet'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Data split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

### Step 2: Start Training

```bash
python models/train.py
```

**Training time:** 4-8 hours (depends on GPU)

**Output:**
```
saved_models/
├── best_model.h5
├── checkpoint_epoch_10.h5
└── training_history.json
```

### Step 3: Monitor Progress

Training will show:
```
Epoch 1/50
━━━━━━━━━━━━━━━━━━━━ 181,000/181,000
- loss: 0.8234 - accuracy: 0.5621
- val_loss: 0.7891 - val_accuracy: 0.5834
```

---

## Evaluate Model

### Step 1: Test Model

```bash
python models/evaluate.py
```

Output:
```
Test Accuracy: 56.3%
Precision (Buy): 61.2%
Recall (Sell): 59.8%
F1-Score: 0.571
```

### Step 2: View Confusion Matrix

The script will generate:
- `confusion_matrix.png`
- `classification_report.txt`
- `evaluation_metrics.json`

### Step 3: Backtest Strategy

```bash
python models/backtest.py
```

Output:
```
Initial Capital: $10,000
Final Capital: $14,235
Total Return: 42.35%
Sharpe Ratio: 1.89
Max Drawdown: -12.3%
```

---

## Visualize Results

### View Sample Images

```bash
python scripts/view_images.py
```

### Plot Training History

```bash
python scripts/plot_history.py
```

---

## Troubleshooting

### Issue: "No module named tensorflow"

```bash
pip install tensorflow
# or for GPU support
pip install tensorflow-gpu
```

### Issue: "Out of Memory"

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

### Issue: "Images not found"

Check paths in `config.py`:
```python
BINANCE_DATA_DIR = "Binance Full History 28gb"  # Correct path
OUTPUT_DIR = "Candlestick_Images_Balanced"
```

### Issue: "Processing too slow"

Reduce number of images per file:
```python
MAX_IMAGES_PER_FILE = 300  # instead of 500
```

---

## Next Steps

1. ✅ **Optimize hyperparameters** (learning rate, batch size, epochs)
2. ✅ **Try different architectures** (ResNet50, VGG16, EfficientNet)
3. ✅ **Implement ensemble methods** (combine multiple models)
4. ✅ **Add technical indicators** (RSI, MACD, Bollinger Bands)
5. ✅ **Deploy for live trading** (with paper trading first!)

---

## Support

- 📖 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/saber-barhoumi/CryptoVisionAI/issues)
- 💬 [Discussions](https://github.com/saber-barhoumi/CryptoVisionAI/discussions)

---

**Happy Training! 🚀**
