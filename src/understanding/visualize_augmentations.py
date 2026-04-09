"""
Visualize different augmentations on a sample leaf.
"""

import matplotlib.pyplot as plt
from prepare_data import prepare_image
from augment import (
    random_rotation, random_flip, random_brightness,
    random_shift, random_zoom, augment_image
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
    # Test on a Parijat leaf
    image_path = "../data/healthy/parijat/WhatsApp Image 2026-03-23 at 10.39.28.jpeg"
    visualize_augmentations(image_path)