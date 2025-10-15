"""
Image Viewer - View generated candlestick images
This script helps you visually inspect the generated images.
"""

import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt

def view_sample_images(image_dir, num_samples=9):
    """
    Display a grid of sample images from each category.
    
    Args:
        image_dir: Path to the Candlestick_Images directory
        num_samples: Number of images to display per category
    """
    image_dir = Path(image_dir)
    
    categories = ['Buy', 'Sell', 'Hold']
    
    for category in categories:
        category_path = image_dir / category
        
        if not category_path.exists():
            print(f"Category {category} not found")
            continue
        
        # Get all images from all pairs in this category
        all_images = []
        for pair_dir in category_path.iterdir():
            if pair_dir.is_dir():
                images = list(pair_dir.glob('*.png'))
                all_images.extend(images)
        
        if len(all_images) == 0:
            print(f"No images found in {category}")
            continue
        
        # Sample random images
        sample_size = min(num_samples, len(all_images))
        sampled_images = random.sample(all_images, sample_size)
        
        # Calculate grid size
        cols = 3
        rows = (sample_size + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        fig.suptitle(f'{category} Signal Images (Sample {sample_size} of {len(all_images)})', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if rows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]
        
        # Display images
        for idx, img_path in enumerate(sampled_images):
            img = Image.open(img_path)
            axes_flat[idx].imshow(img)
            axes_flat[idx].axis('off')
            # Show pair name and timestamp
            title = img_path.parent.name + '\n' + img_path.stem.split('_')[-1]
            axes_flat[idx].set_title(title, fontsize=8)
        
        # Hide empty subplots
        for idx in range(len(sampled_images), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        
        # Save the grid
        output_path = image_dir.parent / f'sample_grid_{category}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample grid for {category}: {output_path}")
        plt.show()
        plt.close()

def show_stats(image_dir):
    """Show statistics about generated images."""
    image_dir = Path(image_dir)
    
    print("=" * 60)
    print("Generated Images Statistics")
    print("=" * 60)
    
    categories = ['Buy', 'Sell', 'Hold']
    total = 0
    
    for category in categories:
        category_path = image_dir / category
        
        if not category_path.exists():
            print(f"{category}: 0 images")
            continue
        
        # Count images per pair
        pair_counts = {}
        for pair_dir in category_path.iterdir():
            if pair_dir.is_dir():
                count = len(list(pair_dir.glob('*.png')))
                if count > 0:
                    pair_counts[pair_dir.name] = count
        
        total_category = sum(pair_counts.values())
        total += total_category
        
        print(f"\n{category}:")
        print(f"  Total: {total_category} images")
        print(f"  Trading pairs: {len(pair_counts)}")
        
        if len(pair_counts) > 0:
            print(f"  Top pairs:")
            for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {pair}: {count} images")
    
    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES: {total}")
    print(f"{'='*60}")

def compare_categories(image_dir):
    """
    Show example images from each category side by side for comparison.
    """
    image_dir = Path(image_dir)
    categories = ['Buy', 'Sell', 'Hold']
    
    # Get one image from each category
    sample_images = {}
    for category in categories:
        category_path = image_dir / category
        if category_path.exists():
            # Get first available image
            for pair_dir in category_path.iterdir():
                if pair_dir.is_dir():
                    images = list(pair_dir.glob('*.png'))
                    if images:
                        sample_images[category] = images[0]
                        break
    
    if len(sample_images) < 3:
        print("Not enough images in all categories for comparison")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparison: Buy vs Sell vs Hold Signals', 
                 fontsize=16, fontweight='bold')
    
    for idx, category in enumerate(categories):
        img = Image.open(sample_images[category])
        axes[idx].imshow(img)
        axes[idx].set_title(category, fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = image_dir.parent / 'comparison_buy_sell_hold.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison image: {output_path}")
    plt.show()
    plt.close()

def main():
    """Main function."""
    print("=" * 60)
    print("Candlestick Image Viewer")
    print("=" * 60)
    
    image_dir = Path(r"c:\Users\saber\Desktop\1trading\Vision Model (CNN)\Candlestick_Images")
    
    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        return
    
    # Show statistics
    show_stats(image_dir)
    
    # Show comparison
    print("\nGenerating comparison image...")
    compare_categories(image_dir)
    
    # View sample images
    print("\nGenerating sample grids...")
    view_sample_images(image_dir, num_samples=9)
    
    print("\n" + "=" * 60)
    print("Done! Check the output images.")
    print("=" * 60)

if __name__ == "__main__":
    main()
