"""
Calculate complete statistics for Binance data and generated images.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

def calculate_binance_data_stats(data_dir):
    """Calculate statistics for all Binance parquet files."""
    data_dir = Path(data_dir)
    parquet_files = list(data_dir.glob('*.parquet'))
    
    print("=" * 80)
    print("BINANCE DATA STATISTICS")
    print("=" * 80)
    
    total_files = len(parquet_files)
    total_rows = 0
    total_size_gb = 0
    file_stats = []
    
    print(f"\nScanning {total_files} parquet files...\n")
    
    for file_path in tqdm(parquet_files, desc="Processing files"):
        try:
            df = pd.read_parquet(file_path)
            rows = len(df)
            size_mb = file_path.stat().st_size / (1024**2)
            
            total_rows += rows
            total_size_gb += size_mb / 1024
            
            file_stats.append({
                'pair': file_path.stem,
                'rows': rows,
                'size_mb': size_mb
            })
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    # Sort by rows
    file_stats.sort(key=lambda x: x['rows'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Files: {total_files:,}")
    print(f"Total Data Points (rows): {total_rows:,}")
    print(f"Total Size: {total_size_gb:.2f} GB")
    print(f"Average rows per file: {total_rows/total_files:,.0f}")
    print(f"Average size per file: {total_size_gb*1024/total_files:.2f} MB")
    
    # Time calculation (assuming 1-minute candles)
    total_minutes = total_rows
    total_hours = total_minutes / 60
    total_days = total_hours / 24
    total_years = total_days / 365
    
    print(f"\n" + "=" * 80)
    print("TIME COVERAGE (assuming 1-minute candles)")
    print("=" * 80)
    print(f"Total minutes: {total_minutes:,}")
    print(f"Total hours: {total_hours:,.0f}")
    print(f"Total days: {total_days:,.0f}")
    print(f"Total years: {total_years:.1f}")
    
    # Group by quote currency
    quote_currencies = {}
    for stat in file_stats:
        pair = stat['pair']
        if '-' in pair:
            quote = pair.split('-')[-1]
            if quote not in quote_currencies:
                quote_currencies[quote] = {'count': 0, 'rows': 0}
            quote_currencies[quote]['count'] += 1
            quote_currencies[quote]['rows'] += stat['rows']
    
    print(f"\n" + "=" * 80)
    print("BY QUOTE CURRENCY")
    print("=" * 80)
    for quote, data in sorted(quote_currencies.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"{quote:10s}: {data['count']:4d} pairs | {data['rows']:,} rows")
    
    # Top 20 pairs by data volume
    print(f"\n" + "=" * 80)
    print("TOP 20 PAIRS BY DATA VOLUME")
    print("=" * 80)
    for i, stat in enumerate(file_stats[:20], 1):
        print(f"{i:2d}. {stat['pair']:20s}: {stat['rows']:,} rows ({stat['size_mb']:.1f} MB)")
    
    # Calculate potential images
    print(f"\n" + "=" * 80)
    print("IMAGE GENERATION POTENTIAL")
    print("=" * 80)
    
    window_size = 30
    future_bars = 5
    
    # Calculate how many images could be generated from each file
    potential_images = 0
    for stat in file_stats:
        rows = stat['rows']
        if rows >= window_size + future_bars:
            potential_images += rows - window_size - future_bars
    
    print(f"Window size: {window_size} candles")
    print(f"Future bars: {future_bars} candles")
    print(f"Potential images from ALL data: {potential_images:,}")
    print(f"  - If 10% sampled: {potential_images//10:,} images")
    print(f"  - If 100 per file: {total_files * 100:,} images")
    print(f"  - If 1000 per file: {total_files * 1000:,} images")
    
    return {
        'total_files': total_files,
        'total_rows': total_rows,
        'total_size_gb': total_size_gb,
        'potential_images': potential_images,
        'quote_currencies': quote_currencies
    }

def calculate_image_stats(image_dir):
    """Calculate statistics for generated images."""
    image_dir = Path(image_dir)
    
    print("\n\n" + "=" * 80)
    print("GENERATED IMAGES STATISTICS")
    print("=" * 80)
    
    categories = ['Buy', 'Sell', 'Hold']
    total_images = 0
    category_stats = {}
    
    for category in categories:
        category_path = image_dir / category
        
        if not category_path.exists():
            print(f"\n{category}: Directory not found")
            category_stats[category] = {'total': 0, 'pairs': {}}
            continue
        
        pair_counts = {}
        total_category = 0
        total_size_mb = 0
        
        for pair_dir in category_path.iterdir():
            if pair_dir.is_dir():
                images = list(pair_dir.glob('*.png'))
                count = len(images)
                if count > 0:
                    pair_counts[pair_dir.name] = count
                    total_category += count
                    
                    # Calculate size
                    for img in images:
                        total_size_mb += img.stat().st_size / (1024**2)
        
        category_stats[category] = {
            'total': total_category,
            'pairs': pair_counts,
            'size_mb': total_size_mb
        }
        
        total_images += total_category
        
        print(f"\n{category.upper()}:")
        print(f"  Total images: {total_category:,}")
        print(f"  Storage: {total_size_mb:.2f} MB")
        print(f"  Trading pairs: {len(pair_counts)}")
        
        if pair_counts:
            print(f"  Top 5 pairs:")
            for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {pair:20s}: {count:4d} images")
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total images: {total_images:,}")
    print(f"Total storage: {sum(s['size_mb'] for s in category_stats.values()):.2f} MB")
    
    if total_images > 0:
        print(f"\nDistribution:")
        for category in categories:
            pct = category_stats[category]['total'] / total_images * 100
            print(f"  {category:6s}: {category_stats[category]['total']:5,} images ({pct:5.1f}%)")
        
        # Average size per image
        avg_size_kb = sum(s['size_mb'] for s in category_stats.values()) * 1024 / total_images
        print(f"\nAverage image size: {avg_size_kb:.2f} KB")
    
    return category_stats

def calculate_processing_estimate(data_stats, current_images):
    """Estimate full processing statistics."""
    print("\n\n" + "=" * 80)
    print("FULL PROCESSING ESTIMATE")
    print("=" * 80)
    
    total_files = data_stats['total_files']
    total_potential = data_stats['potential_images']
    
    # Current rate
    current_files = 10  # We processed 10 files
    current_total = sum(cat['total'] for cat in current_images.values())
    
    if current_total > 0:
        images_per_file = current_total / current_files
        
        print(f"\nBased on current processing:")
        print(f"  Files processed: {current_files}")
        print(f"  Images generated: {current_total:,}")
        print(f"  Rate: {images_per_file:.0f} images per file")
        
        # Estimates for full processing
        print(f"\n" + "-" * 80)
        print("IF PROCESSING ALL {0:,} FILES:".format(total_files))
        print("-" * 80)
        
        estimated_images = total_files * images_per_file
        print(f"Estimated images: {estimated_images:,.0f}")
        
        # Storage estimate
        avg_size_kb = sum(cat['size_mb'] for cat in current_images.values()) * 1024 / current_total
        estimated_storage_gb = (estimated_images * avg_size_kb / 1024) / 1024
        print(f"Estimated storage: {estimated_storage_gb:.2f} GB")
        
        # Time estimate (based on current processing speed)
        # Current: 10 files took about 3.5 minutes
        minutes_per_file = 3.5 / 10
        estimated_time_minutes = total_files * minutes_per_file
        estimated_time_hours = estimated_time_minutes / 60
        
        print(f"Estimated processing time: {estimated_time_hours:.1f} hours ({estimated_time_minutes/60/24:.1f} days)")
        
        # Different sampling strategies
        print(f"\n" + "-" * 80)
        print("SAMPLING STRATEGIES:")
        print("-" * 80)
        
        strategies = [
            (100, "Conservative"),
            (500, "Moderate"),
            (1000, "Aggressive"),
            (None, "Maximum (all windows)")
        ]
        
        for samples, name in strategies:
            if samples is None:
                est = total_potential // 10  # Rough estimate with sampling
                storage = (est * avg_size_kb / 1024) / 1024
                time_hours = total_files * 0.5  # Rough estimate
            else:
                est = total_files * samples
                storage = (est * avg_size_kb / 1024) / 1024
                time_hours = total_files * (samples / images_per_file) * minutes_per_file / 60
            
            print(f"\n{name}:")
            print(f"  Images per file: {samples if samples else 'All'}")
            print(f"  Total images: {est:,}")
            print(f"  Storage: {storage:.2f} GB")
            print(f"  Time: {time_hours:.1f} hours")

def main():
    """Main function."""
    data_dir = Path(r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Binance Full History 28gb")
    image_dir = Path(r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images")
    
    # Calculate Binance data stats
    data_stats = calculate_binance_data_stats(data_dir)
    
    # Calculate image stats
    image_stats = calculate_image_stats(image_dir)
    
    # Calculate processing estimates
    calculate_processing_estimate(data_stats, image_stats)
    
    # Recommendations
    print("\n\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. FOCUS ON MAJOR PAIRS:
   - Process USDT pairs first (272 pairs - most liquid)
   - Then BTC pairs (245 pairs)
   - This gives you ~500 files with best quality data
   
2. BALANCE YOUR DATASET:
   - Current threshold (0.5%) gives 86.8% Hold labels
   - Try threshold = 0.3% for better balance
   - Or filter/sample Hold images after generation
   
3. PROCESSING STRATEGY:
   - Start with 500 images per file for major pairs
   - This gives you ~250,000 images from USDT+BTC pairs
   - Processing time: ~3-4 hours
   - Storage: ~50-70 GB
   
4. FOR FULL DATASET:
   - All 1,000 files × 100 images = 100,000 images
   - Or all 1,000 files × 1000 images = 1,000,000 images
   - Storage: 20-200 GB depending on strategy
   
5. NEXT STEPS:
   - Edit config.py to set desired parameters
   - Run: python data_to_images.py
   - Let it run (can take several hours for full dataset)
   - Monitor disk space
    """)
    
    print("=" * 80)
    print("CALCULATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
