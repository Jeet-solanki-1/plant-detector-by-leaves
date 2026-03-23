import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
# Change to the correct directory (where this script is)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory set to: {os.getcwd()}")

print("Starting leaf preparation...")
print("="*50)

def prepare_image(img_path, target_size=(32, 32), show_steps=False):
    """
    Process one image and optionally show each step
    """
    # STEP 1: Load original image
    img = Image.open(img_path)
    
    # STEP 2: Convert to grayscale
    img_gray = img.convert('L')
    
    # STEP 3: Resize to target size
    img_resized = img_gray.resize(target_size)
    
    # STEP 4: Convert to numpy and normalize
    img_array = np.array(img_resized)
    img_normalized = img_array / 255.0
    
    if show_steps:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(img)
        axes[0].set_title(f"1. Original\n{img.size}")
        axes[0].axis('off')
        
        axes[1].imshow(img_gray, cmap='gray')
        axes[1].set_title("2. Grayscale")
        axes[1].axis('off')
        
        axes[2].imshow(img_resized, cmap='gray')
        axes[2].set_title(f"3. Resized to\n{target_size[0]}x{target_size[1]}")
        axes[2].axis('off')
        
        axes[3].imshow(img_normalized, cmap='gray')
        axes[3].set_title("4. Normalized\n(values 0-1)")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return img_normalized

def load_data(data_folder='data/healthy', target_size=(32, 32)):
    """
    Load all images from your folders:
    - parijat/ -> label 1 (Parijat leaf)
    - mango/ -> label 0 (not Parijat)
    - money-plant/ -> label 0 (not Parijat)
    """
    images = []
    labels = []
    
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    print(f"Looking for data in: {data_folder}")
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"❌ ERROR: Folder '{data_folder}' does not exist!")
        return np.array([]), np.array([])
    
    # Load Parijat leaves (label = 1)
    parijat_folder = os.path.join(data_folder, 'parijat')
    print(f"\n📁 Reading Parijat leaves from: {parijat_folder}")
    
    if os.path.exists(parijat_folder):
        count = 0
        for filename in os.listdir(parijat_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(parijat_folder, filename)
                print(f"  📷 {filename}")
                img_array = prepare_image(img_path, target_size, show_steps=False)
                images.append(img_array)
                labels.append(1)  # 1 = Parijat
                count += 1
        print(f"  ✅ Loaded {count} Parijat leaves (label=1)")
    else:
        print(f"  ❌ Folder not found! Create: {parijat_folder}")
    
    # Load Mango leaves (label = 0 - not Parijat)
    mango_folder = os.path.join(data_folder, 'mango')
    print(f"\n📁 Reading Mango leaves from: {mango_folder}")
    
    if os.path.exists(mango_folder):
        count = 0
        for filename in os.listdir(mango_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(mango_folder, filename)
                print(f"  📷 {filename}")
                img_array = prepare_image(img_path, target_size, show_steps=False)
                images.append(img_array)
                labels.append(0)  # 0 = not Parijat
                count += 1
        print(f"  ✅ Loaded {count} Mango leaves (label=0)")
    else:
        print(f"  ❌ Folder not found! Create: {mango_folder}")
    
    # Load Money-plant leaves (label = 0 - not Parijat)
    money_folder = os.path.join(data_folder, 'money-plant')
    print(f"\n📁 Reading Money-plant leaves from: {money_folder}")
    
    if os.path.exists(money_folder):
        count = 0
        for filename in os.listdir(money_folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(money_folder, filename)
                print(f"  📷 {filename}")
                img_array = prepare_image(img_path, target_size, show_steps=False)
                images.append(img_array)
                labels.append(0)  # 0 = not Parijat
                count += 1
        print(f"  ✅ Loaded {count} Money-plant leaves (label=0)")
    else:
        print(f"  ❌ Folder not found! Create: {money_folder}")
    
    if len(images) == 0:
        print("\n" + "="*50)
        print("⚠️  NO IMAGES FOUND!")
        print("="*50)
        print("\nPlease add your leaf photos to:")
        print("  - data/healthy/parijat/     (Parijat leaves)")
        print("  - data/healthy/mango/       (Other leaves)")
        print("  - data/healthy/money-plant/ (Other leaves)")
        return np.array([]), np.array([])
    
    return np.array(images), np.array(labels)

# Run the data loading
print("\n" + "="*50)
print("STARTING DATA PREPARATION")
print("="*50)

images, labels = load_data('data/healthy', target_size=(32, 32))

if len(images) > 0:
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"✅ Total images loaded: {len(images)}")
    print(f"📊 Images shape: {images.shape}")
    print(f"   This means: {images.shape[0]} images, each {images.shape[1]}x{images.shape[2]} pixels")
    
    # Count how many of each
    parijat_count = sum(labels == 1)
    other_count = sum(labels == 0)
    print(f"\n🌿 Parijat leaves (label=1): {parijat_count}")
    print(f"🍃 Other leaves (label=0): {other_count}")
    
    # Flatten images for neural network
    flattened_images = images.reshape(images.shape[0], -1)
    print(f"\n📐 Flattened shape: {flattened_images.shape}")
    print(f"   Each image is now a list of {flattened_images.shape[1]} numbers (pixel values)")
    
    # Show a grid of processed images
    print("\n" + "="*50)
    print("VISUALIZING PROCESSED IMAGES")
    print("="*50)
    
    # Show first 10 images
    num_to_show = min(10, len(images))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    for i, ax in enumerate(axes.flat):
        if i < num_to_show:
            ax.imshow(images[i], cmap='gray')
            label = "PARIJAT" if labels[i] == 1 else "OTHER"
            color = 'green' if labels[i] == 1 else 'red'
            ax.set_title(f"{label}", color=color)
            ax.axis('off')
    
    plt.suptitle("Your Processed Leaf Images (32x32 grayscale)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Show pixel statistics
    print("\n" + "="*50)
    print("PIXEL STATISTICS")
    print("="*50)
    print(f"📊 Min pixel value: {flattened_images.min():.3f}")
    print(f"📊 Max pixel value: {flattened_images.max():.3f}")
    print(f"📊 Mean pixel value: {flattened_images.mean():.3f}")
    print(f"📊 Standard deviation: {flattened_images.std():.3f}")
    
    print("\n" + "="*50)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*50)
    print("\nYour data is ready for training!")
    print(f"  - {parijat_count} Parijat images")
    print(f"  - {other_count} Other images")
    print(f"  - Total: {len(images)} images")
    
else:
    print("\n❌ No images loaded. Please add photos to the folders:")
    print("  - data/healthy/parijat/")
    print("  - data/healthy/mango/")
    print("  - data/healthy/money-plant/")