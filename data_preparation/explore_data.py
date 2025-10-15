"""
Data Explorer for Binance Trading Data
This script helps you understand the data before generating images.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def explore_single_file(file_path):
    """Explore a single parquet file."""
    try:
        df = pd.read_parquet(file_path)
        
        print(f"\n{'='*60}")
        print(f"File: {Path(file_path).name}")
        print(f"{'='*60}")
        
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nBasic statistics:")
        print(df.describe())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\nMissing values:")
            print(missing[missing > 0])
        else:
            print(f"\nNo missing values")
        
        # Check date range
        if 'timestamp' in df.columns or 'time' in df.columns or 'date' in df.columns:
            time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
            df[time_col] = pd.to_datetime(df[time_col])
            print(f"\nDate range: {df[time_col].min()} to {df[time_col].max()}")
            print(f"Total duration: {df[time_col].max() - df[time_col].min()}")
        
        return df
    
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def explore_directory(data_dir, num_files=5):
    """Explore multiple files in directory."""
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.glob('*.parquet'))
    
    print(f"\n{'='*60}")
    print(f"Directory Overview")
    print(f"{'='*60}")
    print(f"Total parquet files: {len(parquet_files)}")
    
    # Get file sizes
    total_size = sum(f.stat().st_size for f in parquet_files) / (1024**3)  # GB
    print(f"Total size: {total_size:.2f} GB")
    
    # Group by quote currency
    quote_currencies = {}
    for f in parquet_files:
        pair = f.stem
        if '-' in pair:
            quote = pair.split('-')[-1]
            quote_currencies[quote] = quote_currencies.get(quote, 0) + 1
    
    print(f"\nTrading pairs by quote currency:")
    for quote, count in sorted(quote_currencies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {quote}: {count} pairs")
    
    # Sample a few files
    print(f"\nSampling {num_files} files...")
    sample_files = parquet_files[:num_files]
    
    for file_path in sample_files:
        df = explore_single_file(file_path)
        if df is not None and len(df) > 0:
            # Create a small sample plot
            plot_sample_data(df, file_path.stem)

def plot_sample_data(df, pair_name):
    """Plot a sample of the data."""
    try:
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                column_mapping[col] = 'Open'
            elif 'high' in col_lower:
                column_mapping[col] = 'High'
            elif 'low' in col_lower:
                column_mapping[col] = 'Low'
            elif 'close' in col_lower:
                column_mapping[col] = 'Close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'Volume'
        
        df = df.rename(columns=column_mapping)
        
        if 'Close' not in df.columns:
            print(f"No 'Close' column found, skipping plot")
            return
        
        # Plot last 100 candles
        sample_size = min(100, len(df))
        df_sample = df.tail(sample_size)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price plot
        ax1.plot(range(len(df_sample)), df_sample['Close'], label='Close Price', color='blue')
        if 'High' in df_sample.columns and 'Low' in df_sample.columns:
            ax1.fill_between(range(len(df_sample)), 
                            df_sample['Low'], 
                            df_sample['High'], 
                            alpha=0.3, label='High-Low Range')
        ax1.set_title(f'{pair_name} - Last {sample_size} Candles')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        if 'Volume' in df_sample.columns:
            ax2.bar(range(len(df_sample)), df_sample['Volume'], alpha=0.7, color='green')
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Candle Index')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path(r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\data_exploration")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{pair_name}_sample.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Sample plot saved to: {output_dir / f'{pair_name}_sample.png'}")
    
    except Exception as e:
        print(f"Error plotting sample data: {str(e)}")

def analyze_label_distribution(data_dir, num_files=10, window_size=30, 
                               future_bars=5, threshold=2.0):
    """Analyze what the label distribution would be with current settings."""
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.glob('*.parquet'))[:num_files]
    
    print(f"\n{'='*60}")
    print(f"Label Distribution Analysis")
    print(f"{'='*60}")
    print(f"Settings:")
    print(f"  Window Size: {window_size}")
    print(f"  Future Bars: {future_bars}")
    print(f"  Threshold: ±{threshold}%")
    print(f"  Analyzing {len(parquet_files)} files...")
    
    total_labels = {'Buy': 0, 'Sell': 0, 'Hold': 0}
    
    for file_path in tqdm(parquet_files, desc="Analyzing"):
        try:
            df = pd.read_parquet(file_path)
            
            # Standardize column names
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'close' in col_lower:
                    column_mapping[col] = 'Close'
            
            df = df.rename(columns=column_mapping)
            
            if 'Close' not in df.columns or len(df) < window_size + future_bars:
                continue
            
            # Sample every 10th window to speed up analysis
            for idx in range(window_size, len(df) - future_bars, 10):
                current_close = df.iloc[idx]['Close']
                future_close = df.iloc[idx + future_bars]['Close']
                pct_change = ((future_close - current_close) / current_close) * 100
                
                if pct_change >= threshold:
                    total_labels['Buy'] += 1
                elif pct_change <= -threshold:
                    total_labels['Sell'] += 1
                else:
                    total_labels['Hold'] += 1
        
        except Exception as e:
            continue
    
    # Print results
    total = sum(total_labels.values())
    if total > 0:
        print(f"\nEstimated Label Distribution (from {total} samples):")
        print(f"  Buy:  {total_labels['Buy']:6d} ({total_labels['Buy']/total*100:5.1f}%)")
        print(f"  Sell: {total_labels['Sell']:6d} ({total_labels['Sell']/total*100:5.1f}%)")
        print(f"  Hold: {total_labels['Hold']:6d} ({total_labels['Hold']/total*100:5.1f}%)")
        
        # Check if distribution is balanced
        min_pct = min(total_labels.values()) / total * 100
        max_pct = max(total_labels.values()) / total * 100
        
        if max_pct - min_pct > 20:
            print(f"\n⚠️  Warning: Imbalanced dataset!")
            print(f"   Consider adjusting the threshold to balance classes.")
            print(f"   Current range: {min_pct:.1f}% - {max_pct:.1f}%")
        else:
            print(f"\n✓ Dataset appears reasonably balanced")
    else:
        print("\nNo data to analyze")

def main():
    """Main exploration function."""
    print("=" * 60)
    print("Binance Data Explorer")
    print("=" * 60)
    
    data_dir = Path(r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Binance Full History 28gb")
    
    # Explore directory
    explore_directory(data_dir, num_files=3)
    
    # Analyze label distribution
    analyze_label_distribution(
        data_dir=data_dir,
        num_files=10,
        window_size=30,
        future_bars=5,
        threshold=2.0
    )
    
    print("\n" + "=" * 60)
    print("Exploration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
