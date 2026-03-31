"""
Visualize resized images to check if important features are preserved.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List

# ============================================
# CONFIGURATION - CHANGE THIS IF NEEDED
# ============================================

# Path to your data folder (relative to this script)
# Since script is in 'understanding' folder, data is one level up
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "healthy")

# Or absolute path (uncomment and modify if above doesn't work)
# DATA_FOLDER = r"C:\Users\solan\Desktop\github\AI\plant-detector-by-leaves\data\healthy"

TARGET_SIZE = (32, 32)

# ============================================

def load_and_resize_image(
    img_path: str, 
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Tuple[Image.Image, Image.Image]:
    """
    Load image and show original vs resized.
    
    Returns:
        (original_image, resized_image)
    """
    original = Image.open(img_path).convert("RGB")
    resized = original.resize(target_size)
    return original, resized


def visualize_images(
    image_paths: List[str], 
    labels: List[str],
    target_size: Tuple[int, int] = TARGET_SIZE,
    num_samples: int = 12,
    save_path: str = None
) -> None:
    """
    Display original vs resized images side by side.
    """
    n = min(num_samples, len(image_paths))
    image_paths = image_paths[:n]
    labels = labels[:n]
    
    fig, axes = plt.subplots(n, 2, figsize=(8, 2*n))
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        original, resized = load_and_resize_image(img_path, target_size)
        
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"{label}\nOriginal: {original.size}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(resized)
        axes[i, 1].set_title(f"Resized to: {resized.size}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_grid(
    image_paths: List[str],
    labels: List[str],
    target_size: Tuple[int, int] = TARGET_SIZE,
    num_samples: int = 16,
    save_path: str = None
) -> None:
    """
    Display resized images in a grid (no originals).
    """
    n = min(num_samples, len(image_paths))
    image_paths = image_paths[:n]
    labels = labels[:n]
    
    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        _, resized = load_and_resize_image(img_path, target_size)
        axes[i].imshow(resized)
        axes[i].set_title(f"{label}\n{resized.size}", fontsize=9)
        axes[i].axis('off')
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Images as seen by Neural Network ({target_size[0]}x{target_size[1]})", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def analyze_aspect_ratios(image_paths: List[str]) -> None:
    """
    Analyze original aspect ratios to understand what's being cropped.
    """
    if not image_paths:
        print("No images to analyze!")
        return
        
    aspect_ratios = []
    
    for img_path in image_paths:
        img = Image.open(img_path)
        w, h = img.size
        aspect = w / h
        aspect_ratios.append(aspect)
    
    print("\n" + "=" * 50)
    print("ASPECT RATIO ANALYSIS")
    print("=" * 50)
    print(f"Total images: {len(aspect_ratios)}")
    print(f"Min aspect ratio: {min(aspect_ratios):.2f}")
    print(f"Max aspect ratio: {max(aspect_ratios):.2f}")
    print(f"Mean aspect ratio: {np.mean(aspect_ratios):.2f}")
    print(f"Std: {np.std(aspect_ratios):.2f}")
    
    portrait = sum(1 for ar in aspect_ratios if ar < 1)
    landscape = sum(1 for ar in aspect_ratios if ar > 1)
    square = sum(1 for ar in aspect_ratios if 0.95 < ar < 1.05)
    
    print(f"\nPortrait (taller): {portrait} images")
    print(f"Landscape (wider): {landscape} images")
    print(f"Nearly square: {square} images")


def find_images(data_folder: str) -> Tuple[List[str], List[str]]:
    """
    Find all images in the data folder.
    
    Returns:
        (image_paths, labels)
    """
    if not os.path.exists(data_folder):
        print(f"❌ Folder not found: {data_folder}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please check your DATA_FOLDER path.")
        return [], []
    
    classes = ["parijat", "mango", "money-plant"]
    all_paths = []
    all_labels = []
    
    for class_name in classes:
        folder = os.path.join(data_folder, class_name)
        if not os.path.exists(folder):
            continue
            
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(folder, f))
                all_labels.append(class_name)
    
    return all_paths, all_labels


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("IMAGE VISUALIZATION TOOL")
    print("=" * 50)
    
    print(f"\nLooking for images in: {DATA_FOLDER}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Find all images
    image_paths, labels = find_images(DATA_FOLDER)
    
    if not image_paths:
        print("\n❌ No images found!")
        print("\nTrying alternative paths...")
        
        # Try alternative paths
        alt_paths = [
            "data/healthy",
            "../data/healthy",
            "../../data/healthy",
            r"C:\Users\solan\Desktop\github\AI\plant-detector-by-leaves\data\healthy"
        ]
        
        for alt in alt_paths:
            print(f"  Trying: {alt}")
            image_paths, labels = find_images(alt)
            if image_paths:
                DATA_FOLDER = alt
                print(f"✅ Found images in: {DATA_FOLDER}")
                break
    
    if not image_paths:
        print("\n❌ Still no images found. Please check your data folder path.")
        print("\nExpected structure:")
        print("  data/")
        print("    healthy/")
        print("      parijat/")
        print("        image1.jpg")
        print("      mango/")
        print("        image1.jpg")
        print("      money-plant/")
        print("        image1.jpg")
        sys.exit(1)
    
    print(f"\n✅ Found {len(image_paths)} images")
    print(f"   Classes: {set(labels)}")
    
    # Analyze aspect ratios
    analyze_aspect_ratios(image_paths)
    
    # Show original vs resized
    print("\n" + "=" * 50)
    print(f"ORIGINAL VS RESIZED ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
    print("=" * 50)
    visualize_images(image_paths, labels, target_size=TARGET_SIZE, num_samples=8)
    
    # Show grid of resized images
    print("\n" + "=" * 50)
    print("RESIZED GRID (What your network sees)")
    print("=" * 50)
    visualize_grid(image_paths, labels, target_size=TARGET_SIZE, num_samples=16)