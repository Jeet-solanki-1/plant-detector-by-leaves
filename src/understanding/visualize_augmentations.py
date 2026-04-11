"""
Visualize different augmentations on a sample leaf.
"""

import sys
import os

# Add parent directory (src/) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
from prepare_data import prepare_image
from augment import (
    random_rotation,
    random_flip,
    random_brightness,
    random_shift,
    random_zoom,
    augment_image
)

def visualize_augmentations(image_path: str, target_size=(224, 224)):
    """Show original image and various augmentations."""
    
    # Load original
    original = prepare_image(image_path, target_size)
    
    # Create augmentations
    rotated = random_rotation(original, max_angle=15)
    flipped = random_flip(original)
    brightness = random_brightness(original, max_delta=0.2)
    shifted = random_shift(original, max_shift=0.1)
    zoomed = random_zoom(original, max_zoom=0.1)
    augmented = augment_image(original, intensity="medium")
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    transformations = [
        ("Original", original),
        ("Rotated ±15°", rotated),
        ("Flipped", flipped),
        ("Brightness ±20%", brightness),
        ("Shifted", shifted),
        ("Zoomed", zoomed),
    ]
    
    for i, (title, img) in enumerate(transformations):
        ax = axes[i // 3, i % 3]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle("Data Augmentation Examples", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ✅ CORRECTED PATH: two levels up to project root, then data/healthy/parijat/
    # From src/understanding/ -> go up two levels: ../../data/healthy/parijat/
    
    base_path = os.path.join("..", "..", "data", "healthy", "parijat")
    
    # Find any jpeg file in that folder
    if os.path.exists(base_path):
        files = [f for f in os.listdir(base_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
        if files:
            image_path = os.path.join(base_path, files[0])
            print(f"Using image: {image_path}")
            visualize_augmentations(image_path)
        else:
            print(f"No images found in {base_path}")
    else:
        print(f"Path not found: {base_path}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Try alternative: go to project root first
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        alt_path = os.path.join(project_root, "data", "healthy", "parijat")
        print(f"Trying alternative: {alt_path}")
        
        if os.path.exists(alt_path):
            files = [f for f in os.listdir(alt_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
            if files:
                image_path = os.path.join(alt_path, files[0])
                print(f"Found image: {image_path}")
                visualize_augmentations(image_path)