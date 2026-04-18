"""
Multi-Class Leaf Prediction Script.
Predicts Parijat, Mango, or Money-plant with confidence percentages.
"""

import pickle
import numpy as np
from PIL import Image
import sys
import os
from typing import Tuple


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


class VectorizedNeuron:
    def __init__(self, num_inputs: int):
        self.weights = None
        self.bias = 0.0
    
    def forward(self, inputs: np.ndarray) -> float:
        z = np.dot(inputs, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))


def prepare_image(image_path: str, target_size: Tuple[int, int] = (98, 98)) -> np.ndarray:
    """Process image for prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.flatten()


def load_model(filename: str):
    """Load trained multi-class model."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Rebuild hidden layer
    hidden_layer = []
    for weights, bias in zip(data['hidden_weights'], data['hidden_biases']):
        n = VectorizedNeuron(len(weights))
        n.weights = weights
        n.bias = bias
        hidden_layer.append(n)
    
    # Rebuild output layer
    output_layer = []
    for weights, bias in zip(data['output_weights'], data['output_bias']):
        n = VectorizedNeuron(len(weights))
        n.weights = weights
        n.bias = bias
        output_layer.append(n)
    
    class_names = data.get('class_names', ['Parijat', 'Mango', 'Money-plant'])
    
    return hidden_layer, output_layer, class_names


def predict(image_path: str, hidden_layer, output_layer, class_names, target_size=(98, 98)):
    """Predict class for a single image."""
    img_flat = prepare_image(image_path, target_size)
    
    # Forward pass
    hidden_outputs = [n.forward(img_flat) for n in hidden_layer]
    raw_scores = [n.forward(np.array(hidden_outputs)) for n in output_layer]
    
    # Softmax to get probabilities
    probs = softmax(np.array(raw_scores))
    
    predicted_idx = np.argmax(probs)
    predicted_name = class_names[predicted_idx]
    confidence = probs[predicted_idx] * 100
    
    return probs, predicted_idx, predicted_name, confidence


def main():
    print("=" * 60)
    print("🌿 MULTI-CLASS LEAF DETECTOR")
    print("   Identifies: Parijat | Mango | Money-plant")
    print("=" * 60)
    
    # Get image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("\n📸 Enter leaf photo path: ").strip()
    
    img_path = img_path.strip('"').strip("'")
    
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        sys.exit(1)
    
    try:
        # Load model
        model_path = "../traind_models/leaf_model_multi_class_with_Xavier_init_of_weights_20260419_023748.pkl"
        if not os.path.exists(model_path):
            # Try to find latest multi-class model
            import glob
            models = glob.glob("traind_models/leaf_model_multi_class*.pkl")
            if models:
                model_path = max(models, key=os.path.getctime)
                print(f"📂 Using latest model: {os.path.basename(model_path)}")
            else:
                print("❌ No multi-class model found. Train first!")
                sys.exit(1)
        
        print(f"\n📂 Loading model...")
        hidden_layer, output_layer, class_names = load_model(model_path)
        
        # Predict
        probs, predicted_idx, predicted_name, confidence = predict(
            img_path, hidden_layer, output_layer, class_names
        )
        
        # Show results
        print("\n" + "=" * 60)
        print("🌿 PREDICTION RESULTS")
        print("=" * 60)
        
        print(f"\n✅ This is a {predicted_name.upper()} leaf!")
        print(f"   Confidence: {confidence:.1f}%\n")
        
        print("📊 Class Probabilities:")
        print("-" * 40)
        for i, name in enumerate(class_names):
            bar_length = int(probs[i] * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {name:12} | {bar} | {probs[i]*100:5.1f}%")
        
        print("=" * 60)
        
    except FileNotFoundError:
        print("❌ Model file not found. Train the multi-class model first.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()