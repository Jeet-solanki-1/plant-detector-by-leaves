import pickle
import math
import numpy as np
from PIL import Image
import sys

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    def __init__(self, num_inputs):
        self.weights = []
        self.bias = 0.0
    
    def forward(self, inputs):
        z = 0
        for i in range(len(inputs)):
            z += self.weights[i] * inputs[i]
        z += self.bias
        return sigmoid(z)

def prepare_image(image_path, target_size=(32, 32)):
    """
    Process a new leaf image exactly like we did during training
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale
    
    # Resize to 32x32
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Flatten to 1024 numbers
    img_flattened = img_array.flatten()
    
    return img_flattened

def load_model(filename='leaf_model_rgb_v2.pkl'):
    """
    Load the trained model from file
    """
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    # Rebuild hidden layer
    hidden_layer = []
    for weights, bias in zip(model_data['hidden_weights'], model_data['hidden_biases']):
        neuron = Neuron(len(weights))
        neuron.weights = weights
        neuron.bias = bias
        hidden_layer.append(neuron)
    
    # Rebuild output neuron
    output_neuron = Neuron(len(model_data['output_weights']))
    output_neuron.weights = model_data['output_weights']
    output_neuron.bias = model_data['output_bias']
    
    print(f"✅ Model loaded successfully!")
    print(f"   Hidden neurons: {len(hidden_layer)}")
    print(f"   Input size: {len(hidden_layer[0].weights)} pixels")
    
    return hidden_layer, output_neuron

def predict(image_path, hidden_layer, output_neuron):
    """
    Predict if a leaf is Parijat
    """
    # Step 1: Process the image
    print(f"Processing image: {image_path}")
    img_flat = prepare_image(image_path)
    
    # Step 2: Forward pass through hidden layer
    hidden_output = []
    for neuron in hidden_layer:
        hidden_output.append(neuron.forward(img_flat))
    
    # Step 3: Forward pass through output neuron
    output = output_neuron.forward(hidden_output)
    
    # Step 4: Make decision
    prediction = 1 if output > 0.5 else 0
    
    return output, prediction

# ============================================
# Main - Use the model
# ============================================

if __name__ == "__main__":
    print("="*50)
    print("🌿 Parijat Leaf Detector 🌿")
    print("="*50)
    
    # Check if user provided an image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\n📸 Enter the path to your leaf photo:")
        image_path = input("> ").strip()
    
    try:
        # Load the trained model
        print("\n📂 Loading trained model...")
        hidden_layer, output_neuron = load_model('leaf_model_rgb_v2.pkl')
        
        # Make prediction
        print("\n🔍 Analyzing leaf...")
        confidence, prediction = predict(image_path, hidden_layer, output_neuron)
        
        # Show result
        print("\n" + "="*50)
        print("🌿 RESULT 🌿")
        print("="*50)
        
        if prediction == 1:
            print(f"✅ This IS a Parijat leaf!")
            print(f"   Confidence: {confidence*100:.1f}%")
        else:
            print(f"❌ This is NOT a Parijat leaf")
            print(f"   Confidence: {(1-confidence)*100:.1f}%")
        
        print("="*50)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find file '{image_path}'")
        print("   Make sure the path is correct!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you have trained the model first!")