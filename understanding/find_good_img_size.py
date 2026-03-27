"""
Find optimal image size for leaf classification.
Tests different sizes and shows the visual difference.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple

def test_sizes(
    image_path: str,
    sizes: List[Tuple[int, int]],
    title: str = "Leaf Sample"
) -> None:
    """Show the same image at different resolutions."""
    
    original = Image.open(image_path).convert("RGB")
    w, h = original.size
    
    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(3*(len(sizes)+1), 3))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title(f"Original\n{w}×{h}")
    axes[0].axis('off')
    
    # Resized versions
    for i, size in enumerate(sizes):
        resized = original.resize(size)
        axes[i+1].imshow(resized)
        axes[i+1].set_title(f"{size[0]}×{size[1]}\n{size[0]*size[1]} pixels")
        axes[i+1].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_across_sizes(
    image_paths: List[str],
    labels: List[str],
    size: Tuple[int, int]
) -> None:
    """Show multiple images at a specific size."""
    
    cols = 4
    rows = (len(image_paths) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    axes = axes.flatten()
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(img_path).convert("RGB").resize(size)
        axes[i].imshow(img)
        axes[i].set_title(f"{label}\n{size[0]}×{size[1]}", fontsize=9)
        axes[i].axis('off')
    
    for i in range(len(image_paths), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"All Images at {size[0]}×{size[1]}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================
# Experiment
# ============================================

if __name__ == "__main__":
    # Path to your data (adjust as needed)
    data_folder = r"C:\Users\solan\Desktop\github\AI\plant-detector-by-leaves\data\healthy"
    
    # Pick one sample from each class
    samples = []
    labels = []
    
    for class_name in ["parijat", "mango", "money-plant"]:
        folder = os.path.join(data_folder, class_name)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                samples.append(os.path.join(folder, files[0]))
                labels.append(class_name)
    
    if not samples:
        print("No images found!")
        exit(1)
    
    print("=" * 60)
    print("IMAGE SIZE EXPERIMENT")
    print("=" * 60)
    print(f"Testing with: {', '.join(labels)}")
    
    # Test different sizes on one Parijat leaf
    print("\n1️⃣ How size affects detail on a Parijat leaf:")
    sizes = [
           # 4× more pixels
        (96, 96),    # 9× more pixels
        (128, 128),  # 16× more pixels
        
        (152,152),
        (252,252)
    ]
    
    # Find a Parijat leaf
    parijat_folder = os.path.join(data_folder, "parijat")
    if os.path.exists(parijat_folder):
        parijat_files = [f for f in os.listdir(parijat_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if parijat_files:
            test_sizes(os.path.join(parijat_folder, parijat_files[0]), sizes, "Parijat Leaf at Different Sizes")
    
    # Show all three classes at a candidate size
    print("\n2️⃣ How all classes look at 64×64:")
    compare_across_sizes(samples, labels, (64, 64))
    
    print("\n3️⃣ How all classes look at 96×96:")
    compare_across_sizes(samples, labels, (96, 96))
    
    print("\n4️⃣ How all classes look at 128×128:")
    compare_across_sizes(samples, labels, (128, 128))
    compare_across_sizes(samples, labels, (152, 152))
    compare_across_sizes(samples, labels, (252, 252))