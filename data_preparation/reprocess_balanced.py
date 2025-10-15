"""
REBALANCING STRATEGY
====================
Problem: 66.7% Hold (too many neutral signals)
Solution: Lower threshold from 0.3% → 0.15%

Expected NEW Distribution:
- Buy:  28-32%
- Sell: 28-32%
- Hold: 36-44%

This will create a balanced dataset for profitable CNN trading!
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for background processing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NEW OPTIMIZED CONFIGURATION
# ============================================================================

BINANCE_DATA_DIR = r"C:\Users\saber\Desktop\1trading\Vision Model (CNN)\Binance Full History 28gb"
OUTPUT_DIR = r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images_Balanced"

# 🎯 KEY CHANGE: Lower threshold for better balance
PRICE_CHANGE_THRESHOLD = 0.15  # Changed from 0.3% to 0.15%

WINDOW_SIZE = 30
FUTURE_BARS = 5
IMAGE_SIZE = (224, 224)
DPI = 100
MAX_IMAGES_PER_FILE = 500

QUOTE_CURRENCIES = ['USDT', 'BTC']

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_label(data, idx, future_bars=5, threshold=0.15):
    """Calculate label based on future price movement"""
    try:
        current_close = float(data.iloc[idx]['close'])
        if idx + future_bars >= len(data):
            return None
            
        future_close = float(data.iloc[idx + future_bars]['close'])
        price_change = ((future_close - current_close) / current_close) * 100
        
        if price_change > threshold:
            return 'Buy'
        elif price_change < -threshold:
            return 'Sell'
        else:
            return 'Hold'
    except:
        return None

def create_manual_candlestick(data_window, label, output_path, img_count):
    """Create candlestick chart using manual drawing (more reliable)"""
    try:
        # Convert to numeric
        df = data_window.copy()
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        if len(df) < 10:
            return False
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/DPI, IMAGE_SIZE[1]/DPI), dpi=DPI)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Get price range for scaling
        all_highs = df['high'].values
        all_lows = df['low'].values
        price_min = np.min(all_lows)
        price_max = np.max(all_highs)
        price_range = price_max - price_min
        
        if price_range == 0:
            plt.close(fig)
            return False
        
        # Add padding
        padding = price_range * 0.1
        y_min = price_min - padding
        y_max = price_max + padding
        
        # Candlestick parameters
        candle_width = 0.6
        
        # Draw each candlestick
        for i, (idx, row) in enumerate(df.iterrows()):
            open_price = float(row['open'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            close_price = float(row['close'])
            
            # Determine color
            color = '#00ff00' if close_price >= open_price else '#ff0000'
            
            # Draw high-low line (wick)
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=1, solid_capstyle='round')
            
            # Draw open-close rectangle (body)
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            if body_height > 0:
                rect = patches.Rectangle(
                    (i - candle_width/2, body_bottom),
                    candle_width,
                    body_height,
                    linewidth=0,
                    edgecolor=color,
                    facecolor=color
                )
                ax.add_patch(rect)
            else:
                # Doji candle (open == close)
                ax.plot([i - candle_width/2, i + candle_width/2], 
                       [open_price, open_price], 
                       color=color, linewidth=1.5)
        
        # Set limits and remove axes
        ax.set_xlim(-0.5, len(df) - 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        
        # Save
        pair_name = output_path.name.split('_')[0] if '_' in output_path.name else 'unknown'
        filename = f"{pair_name}_{img_count:06d}.png"
        filepath = output_path / label / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(
            filepath,
            dpi=DPI,
            bbox_inches='tight',
            pad_inches=0.05,
            facecolor='black'
        )
        plt.close(fig)
        
        return True
        
    except Exception as e:
        plt.close('all')
        return False

def process_file(file_path, output_dir, max_images=500):
    """Process single parquet file"""
    try:
        df = pd.read_parquet(file_path)
        
        if len(df) < WINDOW_SIZE + FUTURE_BARS:
            return {'Buy': 0, 'Sell': 0, 'Hold': 0}
        
        counts = {'Buy': 0, 'Sell': 0, 'Hold': 0}
        total_images = 0
        
        # Sample indices to get max_images
        total_possible = len(df) - WINDOW_SIZE - FUTURE_BARS
        if total_possible <= 0:
            return counts
            
        # Use evenly spaced samples for diversity
        if total_possible > max_images:
            step = total_possible // max_images
            indices = range(0, total_possible, step)
        else:
            indices = range(total_possible)
        
        for idx in indices:
            if total_images >= max_images:
                break
                
            # Get label
            label = calculate_label(df, idx, FUTURE_BARS, PRICE_CHANGE_THRESHOLD)
            if label is None:
                continue
            
            # Get window
            window = df.iloc[idx:idx + WINDOW_SIZE]
            
            # Create image
            pair_name = file_path.stem
            if create_manual_candlestick(window, label, output_dir / pair_name, counts[label]):
                counts[label] += 1
                total_images += 1
        
        return counts
        
    except Exception as e:
        return {'Buy': 0, 'Sell': 0, 'Hold': 0}

# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("\n" + "="*80)
    print("REBALANCING DATASET - OPTIMIZED THRESHOLD")
    print("="*80)
    print(f"OLD Threshold: 0.3% → NEW Threshold: {PRICE_CHANGE_THRESHOLD}%")
    print(f"Expected Distribution: ~30% Buy, ~30% Sell, ~40% Hold")
    print(f"Window Size: {WINDOW_SIZE} candles")
    print(f"Future Bars: {FUTURE_BARS}")
    print(f"Image Size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Max Images/File: {MAX_IMAGES_PER_FILE}")
    print("="*80 + "\n")
    
    # Setup
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for label in ['Buy', 'Sell', 'Hold']:
        (output_path / label).mkdir(exist_ok=True)
    
    # Get files
    data_path = Path(BINANCE_DATA_DIR)
    all_files = list(data_path.glob("*.parquet"))
    
    # Filter by quote currency
    files_to_process = [
        f for f in all_files 
        if any(f.stem.endswith(quote) for quote in QUOTE_CURRENCIES)
    ]
    
    print(f"Files found: {len(files_to_process)}")
    print(f"Quote currencies: {', '.join(QUOTE_CURRENCIES)}")
    print(f"Processing with {PRICE_CHANGE_THRESHOLD}% threshold...\n")
    
    # Process
    total_counts = {'Buy': 0, 'Sell': 0, 'Hold': 0}
    processed_files = 0
    
    for file in tqdm(files_to_process, desc="Processing"):
        counts = process_file(file, output_path, MAX_IMAGES_PER_FILE)
        for key in total_counts:
            total_counts[key] += counts[key]
        
        processed_files += 1
        
        # Progress update every 50 files
        if processed_files % 50 == 0:
            total = sum(total_counts.values())
            if total > 0:
                print(f"\nProgress Update:")
                print(f"  Files: {processed_files}")
                print(f"  Images: {total:,} (Buy: {total_counts['Buy']/total*100:.1f}% | "
                      f"Sell: {total_counts['Sell']/total*100:.1f}% | "
                      f"Hold: {total_counts['Hold']/total*100:.1f}%)")
    
    # Results
    total_images = sum(total_counts.values())
    
    print("\n" + "="*80)
    print("REBALANCING COMPLETE!")
    print("="*80)
    print(f"Files Processed: {processed_files}")
    print(f"Total Images: {total_images:,}")
    print(f"  - Buy:  {total_counts['Buy']:,} ({total_counts['Buy']/total_images*100:.1f}%)")
    print(f"  - Sell: {total_counts['Sell']:,} ({total_counts['Sell']/total_images*100:.1f}%)")
    print(f"  - Hold: {total_counts['Hold']:,} ({total_counts['Hold']/total_images*100:.1f}%)")
    print(f"\nOutput: {OUTPUT_DIR}")
    print("="*80)
    
    # Calculate improvement
    old_buy_pct = 16.4
    old_sell_pct = 16.9
    old_hold_pct = 66.7
    
    new_buy_pct = total_counts['Buy']/total_images*100
    new_sell_pct = total_counts['Sell']/total_images*100
    new_hold_pct = total_counts['Hold']/total_images*100
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    print(f"Buy:  {old_buy_pct:.1f}% → {new_buy_pct:.1f}% ({new_buy_pct - old_buy_pct:+.1f}%)")
    print(f"Sell: {old_sell_pct:.1f}% → {new_sell_pct:.1f}% ({new_sell_pct - old_sell_pct:+.1f}%)")
    print(f"Hold: {old_hold_pct:.1f}% → {new_hold_pct:.1f}% ({new_hold_pct - old_hold_pct:+.1f}%)")
    print("="*80)
    
    # Save stats
    stats = {
        'total_images': total_images,
        'distribution': total_counts,
        'percentages': {
            'buy': new_buy_pct,
            'sell': new_sell_pct,
            'hold': new_hold_pct
        },
        'threshold': PRICE_CHANGE_THRESHOLD,
        'window_size': WINDOW_SIZE,
        'future_bars': FUTURE_BARS,
        'files_processed': processed_files,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'rebalanced_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats saved to: {output_path / 'rebalanced_stats.json'}")

if __name__ == "__main__":
    main()
