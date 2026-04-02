"""
Data preparation module for leaf classification.
Loads and preprocesses images from folder structure.
"""

import os
from PIL import Image
import numpy as np 
from typing import Tuple, List, Optional
from datetime import datetime

def prepare_image(
    img_path:str,
    target_size:Tuple[int,int]
    ) -> np.ndarray:
    """
    Load and preprocess a single image.

    Args:
        img_path: Path to image file
        target_size: Desired output size (height, width)

    Return:
        Normalized image array with values in [0,1], shape (H, W, 3)
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)/255.0
    return img_array

def load_data(
    data_folder:str,
    target_size:Tuple[int,int],
    verbose: bool = False
    ) -> Tuple[np.ndarray,np.ndarray]:
    """
    Load all images from subfolders with their labels.

    Folder structure expected:
        data_folder/
            parijat/    -> label 1
            mango/  -> label 0
            money-plant/ -> label 0

    Args:
        data_folder: Root folder contaning class subfolders
        target_size: Desired image size
        verboseL: If True, print loading progress (default False)
    Returns:
        Tuple of (images array, labels array)
            images shape: (N, H, W, 3)
            labels shape: (N,)
    """
    images: List[np.ndarray] = []
    labels: List[int] = []
    class_mapping = {
        "parijat": 1,
        "mango": 0,
        "money-plant":0
    }

    for class_name,label in class_mapping.items():
        class_folder = os.path.join(data_folder,class_name)
        if not os.path.exists(class_folder):
            if verbose:
                print(f"Folder not found: {class_folder}")
            continue
        if verbose:
            print(f"Loading {class_name}...")

        count = 0
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png','.jpeg','.jpg')):
                img_path = os.path.join(class_folder,filename)
                try:
                    img_array=prepare_image(img_path,target_size)
                    images.append(img_array)
                    labels.append(label)
                    count+=1
                except Exception:
                    # Silent skip on errors (production)
                    continue
        if verbose:
            print(f"Loaded {count} images from {class_name}")

    if verbose:
        print(f"\n Total images loaded: {len(images)}")

    return np.array(images,dtype=np.float32), np.array(labels, dtype=np.int32)

def visualize_samples(
    images: np.ndarray, 
    labels: np.ndarray, 
    num_samples: int = 9,
    save_path: Optional[str] = None
) -> None:
    """
    Display sample images with their labels.
    Only imports matplotlib when called (lazy loading).
    
    Args:
        images: Array of images
        labels: Array of labels
        num_samples: Number of samples to display
        save_path: If provided, save figure to path instead of showing
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        if i < num_samples and i < len(images):
            ax.imshow(images[i])
            ax.set_title("Parijat" if labels[i] == 1 else "Other")
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("visuals", exist_ok=True)
    images, labels = load_data("data/healthy",target_size=(98,98),verbose=True)
    print(f"images: {images.shape}, labels: {labels.shape}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"visuals/samples_{timestamp}.png"
    visualize_samples(images, labels, save_path=save_path)