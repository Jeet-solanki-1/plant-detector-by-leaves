"""
Prediction module for leaf classification.
Loads trained model and predicts on new images.
"""

import pickle
import numpy as np
from PIL import Image
import sys
import os
from typing import Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


class VectorizedNeuron:
    """Neuron for prediction (weights loaded from trained model)."""
    
    def __init__(self, num_inputs: int):
        self.weights = None
        self.bias = 0.0
    
    def forward(self, inputs: np.ndarray) -> float:
        """Forward pass: z = w·x + b, output = sigmoid(z)."""
        z = np.dot(inputs, self.weights) + self.bias
        return sigmoid(z)


def prepare_image(image_path: str, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Process image for prediction.
    
    Args:
        image_path: Path to image file
        target_size: Desired size (height, width)
    
    Returns:
        Flattened normalized image array
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array.flatten()


def load_model(filename: str = "leaf_model_refactored.pkl") -> Tuple[list, VectorizedNeuron]:
    """
    Load trained model from file.
    
    Args:
        filename: Path to saved model (.pkl)
    
    Returns:
        Tuple of (hidden_layer list, output_neuron)
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Rebuild hidden layer
    hidden_layer = []
    for weights, bias in zip(data['hidden_weights'], data['hidden_biases']):
        n = VectorizedNeuron(len(weights))
        n.weights = weights
        n.bias = bias
        hidden_layer.append(n)
    
    # Rebuild output neuron
    output = VectorizedNeuron(len(data['output_weights']))
    output.weights = data['output_weights']
    output.bias = data['output_bias']
    
    return hidden_layer, output


def predict(
    image_path: str, 
    hidden_layer: list, 
    output_neuron: VectorizedNeuron,
    target_size:Tuple[int,int]=(64,64)
) -> Tuple[float, int]:
    """
    Predict if leaf is Parijat.
    
    Args:
        image_path: Path to leaf image
        hidden_layer: Trained hidden layer neurons
        output_neuron: Trained output neuron
    
    Returns:
        Tuple of (confidence, prediction)
            confidence: raw output (0-1)
            prediction: 1 = Parijat, 0 = Other
    """
    img_flat = prepare_image(image_path,target_size=(64,64))
    hidden_outputs = [n.forward(img_flat) for n in hidden_layer]
    final_output = output_neuron.forward(np.array(hidden_outputs))
    return final_output, 1 if final_output > 0.5 else 0


def main():
    print("=" * 50)
    print("Parijat Leaf Detector")
    print("=" * 50)
    
    # Get image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("\nEnter leaf photo path: ").strip()
    
    img_path = img_path.strip('"').strip("'")
    
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)
    
    try:
        # Load model
        hidden, output = load_model(filename="traind_models/leaf_model_refactored_v1_rgb_64i_70h_20260327_205949.pkl")
        
        # Predict
        confidence, prediction = predict(img_path, hidden, output,target_size=(64,64))
        
        # Show result
        print("\n" + "=" * 50)
        if prediction == 1:
            print(f"✅ PARIJAT LEAF")
            print(f"   Confidence: {confidence*100:.1f}%")
        else:
            print(f"❌ NOT PARIJAT")
            print(f"   Confidence: {(1-confidence)*100:.1f}%")
        print("=" * 50)
        
    except FileNotFoundError:
        print("Model file not found. Train the model first.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()