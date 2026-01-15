# ðŸš€ CryptoVisionAI

**AI-Powered Cryptocurrency Trading Using Computer Vision**

Transform candlestick charts into profitable trading signals using Convolutional Neural Networks (CNN).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Binance%20Full%20History-20BEFF)](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)

---

## ðŸŽ¯ Project Overview

This project converts **Binance trading data** from Kaggle into **candlestick chart images** and trains a **CNN model** to predict:
- ðŸ“ˆ **BUY** signals (bullish patterns)
- ðŸ“‰ **SELL** signals (bearish patterns) 
- â¸ï¸ **HOLD** signals (neutral patterns)

### ðŸ”¥ Key Features

- âœ… **1.5 Billion Trading Candles** analyzed
- âœ… **260,000+ Candlestick Images** generated
- âœ… **Balanced Dataset** (26.5% Buy, 26.9% Sell, 46.6% Hold)
- âœ… **517 USDT & BTC Pairs** processed
- âœ… **0.15% Threshold** for optimal signal detection
- âœ… **30-Candle Window** for pattern recognition
- âœ… **5-Bar Future Prediction** for trading decisions

---

## ðŸ“Š Dataset

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

### ðŸ“ˆ Label Distribution (Balanced)

```
Buy:  26.5% (67,533 images)  âœ… Optimal
Sell: 26.9% (68,328 images)  âœ… Optimal
Hold: 46.6% (118,563 images) âœ… Controlled
```

**Improvement over naive threshold:**
- Old (0.3%): 16% Buy / 17% Sell / 67% Hold âŒ
- New (0.15%): 27% Buy / 27% Sell / 47% Hold âœ…
- **+60% more tradeable signals!**

---

## ðŸ› ï¸ Installation

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

## ðŸš€ Quick Start

### 1ï¸âƒ£ **Download Binance Data**

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

### 2ï¸âƒ£ **Generate Candlestick Images**

```bash
python data_preparation/reprocess_balanced.py
```

This will:
- Process 517 USDT & BTC trading pairs
- Generate ~260,000 images (takes ~13 hours)
- Apply 0.15% threshold for balanced labels
- Save to `Candlestick_Images_Balanced/`

### 3ï¸âƒ£ **Train CNN Model**

```bash
python models/train.py
```

### 4ï¸âƒ£ **Evaluate & Backtest**

```bash
python models/evaluate.py
```

---

## ðŸ“ Project Structure

```
CryptoVisionAI/
â”‚
â”œâ”€â”€ data_preparation/          # Data processing scripts
â”‚   â”œâ”€â”€ explore_data.py        # Analyze Binance data
â”‚   â”œâ”€â”€ calculate_stats.py     # Dataset statistics
â”‚   â””â”€â”€ reprocess_balanced.py  # Generate balanced images
â”‚
â”œâ”€â”€ models/                    # CNN models
â”‚   â”œâ”€â”€ cnn_model.py           # Model architecture
â”‚   â”œâ”€â”€ train.py               # Training pipeline
â”‚   â””â”€â”€ evaluate.py            # Evaluation & backtesting
â”‚
â”œâ”€â”€ utils/                     # Utility functions
â”‚   â”œâ”€â”€ chart_generator.py     # Chart creation
â”‚   â””â”€â”€ label_calculator.py    # Label generation
â”‚
â”œâ”€â”€ scripts/                   # Helper scripts
â”‚   â”œâ”€â”€ check_balance.ps1      # Dataset balance checker
â”‚   â””â”€â”€ view_images.py         # Visualize samples
â”‚
â”œâ”€â”€ docs/                      # Documentation
â”‚   â”œâ”€â”€ QUICKSTART.md          # Setup guide
â”‚   â””â”€â”€ DATA_ANALYSIS_RESULTS.md  # Analysis results
â”‚
â”œâ”€â”€ config.py                  # Configuration
â”œâ”€â”€ requirements.txt           # Dependencies
â”œâ”€â”€ .gitignore                 # Ignore large files
â””â”€â”€ README.md                  # This file
```

---

## âš™ï¸ Configuration

Edit `config.py` to customize:

```python
# Data Processing
WINDOW_SIZE = 30            # Candles per image
FUTURE_BARS = 5             # Prediction horizon
PRICE_THRESHOLD = 0.15      # Â±0.15% for Buy/Sell

# Image Settings
IMAGE_SIZE = (224, 224)     # CNN input size
MAX_IMAGES_PER_FILE = 500   # Limit per pair

# Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

---

## ðŸ§  CNN Architecture

### Custom Architecture

```
Input (224x224x3)
    â†“
Conv2D(32) â†’ ReLU â†’ MaxPool
    â†“
Conv2D(64) â†’ ReLU â†’ MaxPool
    â†“
Conv2D(128) â†’ ReLU â†’ MaxPool
    â†“
Flatten â†’ Dense(256) â†’ Dropout(0.5)
    â†“
Output (3 classes: Buy/Sell/Hold)
```

### Transfer Learning Options

- **ResNet50** (Recommended)
- **VGG16**
- **EfficientNet**

---

## ðŸ“ˆ Methodology

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

## ðŸŽ¯ Expected Performance

| Metric | Target |
|--------|--------|
| **Accuracy** | >55% |
| **Precision (Buy)** | >60% |
| **Recall (Sell)** | >60% |
| **F1-Score** | >0.55 |
| **Sharpe Ratio** | >1.5 |

---

## ðŸ› ï¸ Tools & Technologies

- **Python 3.9**
- **TensorFlow / PyTorch** (Deep Learning)
- **Pandas** (Data manipulation)
- **mplfinance** (Chart generation)
- **NumPy** (Numerical computing)
- **Matplotlib** (Visualization)
- **scikit-learn** (Model evaluation)

---

## ðŸ“š Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Data Analysis Results](docs/DATA_ANALYSIS_RESULTS.md)
- [Kaggle Dataset](https://www.kaggle.com/datasets/jorijnsmit/binance-full-history)

---

## ðŸ¤ Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ðŸ“ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## âš ï¸ Disclaimer

**This is for educational purposes only.**

- âŒ Not financial advice
- âš ï¸ Trading cryptocurrencies carries significant risk
- ðŸ“‰ Past performance â‰  future results
- ðŸ“š Always do your own research (DYOR)

---

## ðŸ‘¤ Author

**Saber Barhoumi**

- GitHub: [@saber-barhoumi](https://github.com/saber-barhoumi)
- Project: [CryptoVisionAI](https://github.com/saber-barhoumi/CryptoVisionAI)

---

## ðŸ™ Acknowledgments

- [Binance](https://www.binance.com/) for historical data
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [Jorijn Smit](https://www.kaggle.com/jorijnsmit) for creating the dataset
- TensorFlow/PyTorch teams
- Open-source ML community

---

## ðŸ“ž Support

â­ **Star this repository** if you found it helpful!

ðŸ“ [Report Issues](https://github.com/saber-barhoumi/CryptoVisionAI/issues)

ðŸ’¬ [Start Discussion](https://github.com/saber-barhoumi/CryptoVisionAI/discussions)

---

**Built with â¤ï¸ for the crypto trading community**

<!-- Last updated: 2026-01-15 09:36:33 -->