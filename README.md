# 🚀 CryptoVisionAI

**AI-Powered Cryptocurrency Trading Using Computer Vision**

Transform candlestick charts into profitable trading signals using Convolutional Neural Networks (CNN).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Binance%20Full%20History-20BEFF)](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)

---

## 🎯 Project Overview

This project converts **Binance trading data** from Kaggle into **candlestick chart images** and trains a **CNN model** to predict:
- 📈 **BUY** signals (bullish patterns)
- 📉 **SELL** signals (bearish patterns) 
- ⏸️ **HOLD** signals (neutral patterns)

### 🔥 Key Features

- ✅ **1.5 Billion Trading Candles** analyzed
- ✅ **260,000+ Candlestick Images** generated
- ✅ **Balanced Dataset** (26.5% Buy, 26.9% Sell, 46.6% Hold)
- ✅ **517 USDT & BTC Pairs** processed
- ✅ **0.15% Threshold** for optimal signal detection
- ✅ **30-Candle Window** for pattern recognition
- ✅ **5-Bar Future Prediction** for trading decisions

---

## 📊 Dataset

### Raw Data Source
**Kaggle Dataset**: [Binance Full History](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)

| Metric | Value |
|--------|-------|
| **Trading Pairs** | 1,000 |
| **Total Candles** | 1.5 Billion |
| **Size** | 32.22 GB |
| **Format** | Parquet |
| **Timeframe** | 1-minute |
| **Period** | 2017 - 2021+ |

### Processed Images

| Metric | Value |
|--------|-------|
| **Images Generated** | 260,000+ |
| **Resolution** | 224x224 |
| **Pairs Processed** | 517 (USDT + BTC) |
| **Training Split** | 70% / 15% / 15% |

### 📈 Label Distribution (Balanced)

```
Buy:  26.5% (67,533 images)  ✅ Optimal
Sell: 26.9% (68,328 images)  ✅ Optimal
Hold: 46.6% (118,563 images) ✅ Controlled
```

**Improvement over naive threshold:**
- Old (0.3%): 16% Buy / 17% Sell / 67% Hold ❌
- New (0.15%): 27% Buy / 27% Sell / 47% Hold ✅
- **+60% more tradeable signals!**

---

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
pip
Git
```

### Clone Repository

```bash
git clone https://github.com/saber-barhoumi/CryptoVisionAI.git
cd CryptoVisionAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ **Download Binance Data**

Download from Kaggle: [Binance Full History](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)

**Using Kaggle CLI (Recommended):**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (32GB)
kaggle datasets download -d jorijnsmit/binance-full-history

# Extract
unzip binance-full-history.zip -d "Binance Full History 28gb"
```

**Or manual download:**
1. Visit [dataset page](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)
2. Click "Download" (requires Kaggle account)
3. Extract to `Binance Full History 28gb/` folder

### 2️⃣ **Generate Candlestick Images**

```bash
python data_preparation/reprocess_balanced.py
```

This will:
- Process 517 USDT & BTC trading pairs
- Generate ~260,000 images (takes ~13 hours)
- Apply 0.15% threshold for balanced labels
- Save to `Candlestick_Images_Balanced/`

### 3️⃣ **Train CNN Model**

```bash
python models/train.py
```

### 4️⃣ **Evaluate & Backtest**

```bash
python models/evaluate.py
```

---

## 📁 Project Structure

```
CryptoVisionAI/
│
├── data_preparation/          # Data processing scripts
│   ├── explore_data.py        # Analyze Binance data
│   ├── calculate_stats.py     # Dataset statistics
│   └── reprocess_balanced.py  # Generate balanced images
│
├── models/                    # CNN models
│   ├── cnn_model.py           # Model architecture
│   ├── train.py               # Training pipeline
│   └── evaluate.py            # Evaluation & backtesting
│
├── utils/                     # Utility functions
│   ├── chart_generator.py     # Chart creation
│   └── label_calculator.py    # Label generation
│
├── scripts/                   # Helper scripts
│   ├── check_balance.ps1      # Dataset balance checker
│   └── view_images.py         # Visualize samples
│
├── docs/                      # Documentation
│   ├── QUICKSTART.md          # Setup guide
│   └── DATA_ANALYSIS_RESULTS.md  # Analysis results
│
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignore large files
└── README.md                  # This file
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Data Processing
WINDOW_SIZE = 30            # Candles per image
FUTURE_BARS = 5             # Prediction horizon
PRICE_THRESHOLD = 0.15      # ±0.15% for Buy/Sell

# Image Settings
IMAGE_SIZE = (224, 224)     # CNN input size
MAX_IMAGES_PER_FILE = 500   # Limit per pair

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

---

## 🧠 CNN Architecture

### Custom Architecture

```
Input (224x224x3)
    ↓
Conv2D(32) → ReLU → MaxPool
    ↓
Conv2D(64) → ReLU → MaxPool
    ↓
Conv2D(128) → ReLU → MaxPool
    ↓
Flatten → Dense(256) → Dropout(0.5)
    ↓
Output (3 classes: Buy/Sell/Hold)
```

### Transfer Learning Options

- **ResNet50** (Recommended)
- **VGG16**
- **EfficientNet**

---

## 📈 Methodology

### Data Preparation
1. Load Binance OHLCV data from Kaggle (Parquet format)
2. Create 30-candle sliding windows
3. Generate candlestick chart images (224x224)
4. Calculate labels based on 5-bar future returns
5. Apply 0.15% threshold for signal detection

### Labeling Strategy
```python
future_return = (close[t+5] - close[t]) / close[t] * 100

if future_return > 0.15%:
    label = BUY    # Price will rise
elif future_return < -0.15%:
    label = SELL   # Price will fall
else:
    label = HOLD   # Price stays neutral
```

### Training Pipeline
1. Split data: 70% train / 15% val / 15% test
2. Data augmentation: rotation, zoom, brightness
3. Train CNN with class weights
4. Early stopping & learning rate reduction
5. Save best model checkpoint

---

## 🎯 Expected Performance

| Metric | Target |
|--------|--------|
| **Accuracy** | >55% |
| **Precision (Buy)** | >60% |
| **Recall (Sell)** | >60% |
| **F1-Score** | >0.55 |
| **Sharpe Ratio** | >1.5 |

---

## 🛠️ Tools & Technologies

- **Python 3.9**
- **TensorFlow / PyTorch** (Deep Learning)
- **Pandas** (Data manipulation)
- **mplfinance** (Chart generation)
- **NumPy** (Numerical computing)
- **Matplotlib** (Visualization)
- **scikit-learn** (Model evaluation)

---

## 📚 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Data Analysis Results](docs/DATA_ANALYSIS_RESULTS.md)
- [Kaggle Dataset](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This is for educational purposes only.**

- ❌ Not financial advice
- ⚠️ Trading cryptocurrencies carries significant risk
- 📉 Past performance ≠ future results
- 📚 Always do your own research (DYOR)

---

## 👤 Author

**Saber Barhoumi**

- GitHub: [@saber-barhoumi](https://github.com/saber-barhoumi)
- Project: [CryptoVisionAI](https://github.com/saber-barhoumi/CryptoVisionAI)

---

## 🙏 Acknowledgments

- [Binance](https://www.binance.com/) for historical data
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [Jorijn Smit](https://www.kaggle.com/jorijnsmit) for creating the dataset
- TensorFlow/PyTorch teams
- Open-source ML community

---

## 📞 Support

⭐ **Star this repository** if you found it helpful!

📝 [Report Issues](https://github.com/saber-barhoumi/CryptoVisionAI/issues)

💬 [Start Discussion](https://github.com/saber-barhoumi/CryptoVisionAI/discussions)

---

**Built with ❤️ for the crypto trading community**

<!-- Last updated: 2025-10-25 16:03:04 -->