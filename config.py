"""
Configuration file for candlestick image generation and CNN training.
Modify these parameters to customize the behavior.
"""

# ====================
# Data Processing Config
# ====================

# Window size: Number of candles to include in each image
WINDOW_SIZE = 30  # Try: 20, 30, 50, or 100

# Future bars: How many bars ahead to check for price movement
FUTURE_BARS = 5  # Try: 3, 5, 10

# Threshold: Percentage change for Buy/Sell classification
# Based on data exploration, 2.0% gives 99.5% Hold labels
# 0.5% gives much better balance for minute-by-minute data
THRESHOLD_PERCENT = 0.5  # Try: 0.3, 0.5, 0.7, 1.0

# ====================
# Image Config
# ====================

# Image dimensions (width, height)
IMAGE_WIDTH = 224  # Standard for CNNs: 224, 256
IMAGE_HEIGHT = 224

# Image resolution
DPI = 100

# ====================
# Processing Config
# ====================

# Maximum files to process (None for all)
MAX_FILES = None  # Set to 10 for testing, None for all files

# Maximum images per file (None for all possible windows)
MAX_IMAGES_PER_FILE = 100  # Set to 50-100 for testing, None for all

# ====================
# Paths
# ====================

# Data directory containing parquet files
DATA_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Binance Full History 28gb"

# Output directory for images
OUTPUT_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images"

# ====================
# Label Strategy
# ====================

# Label types:
# 'simple': Buy if price goes up by threshold, Sell if down, Hold otherwise
# 'strict': Buy only for strong upward movement, Sell for strong downward
LABEL_STRATEGY = 'simple'

# For 'strict' strategy, you can define different thresholds
BUY_THRESHOLD = 2.0   # Minimum % increase for Buy
SELL_THRESHOLD = -2.0  # Maximum % decrease for Sell
# Anything in between is Hold

# ====================
# Trading Pairs Selection
# ====================

# Filter specific trading pairs (empty list = process all)
SELECTED_PAIRS = [
    # Examples:
    # 'BTC-USDT',
    # 'ETH-USDT',
    # 'BNB-USDT',
]

# Filter by quote currency (empty list = all)
QUOTE_CURRENCIES = ['USDT']  # e.g., ['USDT', 'BUSD', 'BTC']

# ====================
# Data Quality
# ====================

# Minimum number of data points required in a file
MIN_DATA_POINTS = 1000

# Remove outliers (very large price movements that might be errors)
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 50  # % change considered as outlier
