from prepare_data import load_data
import numpy as np
import math
import random

print("="*50)
print("training begins")
print("="*50)

# Load data
images, labels = load_data('data/healthy')
flattened_images = images.reshape(images.shape[0], -1)

print(f"Loaded {len(flattened_images)} images")
print(f"Parijat: {sum(labels)} | Other: {len(labels)-sum(labels)}")

# Shuffle the data (important!)
indices = np.random.permutation(len(flattened_images))
flattened_images = flattened_images[indices]
labels = labels[indices]

# Split into train (80%) and test (20%)
split = int(0.8 * len(flattened_images))
train_images = flattened_images[:split]
train_labels = labels[:split]
test_images = flattened_images[split:]
test_labels = labels[split:]

print(f"Training: {len(train_images)} images")
print(f"Testing: {len(test_images)} images")

def sigmoid(x):
    return 1/(1+math.exp(-x))

class Neuron():
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = 0.0
    
    def forward(self, inputs):
        z = 0
        for i in range(len(inputs)):
            z += self.weights[i] * inputs[i]
        z += self.bias
        return sigmoid(z)

# Network architecture
input_size = 32 * 32  # 1024
hidden_size = 5  # You can try smaller like 15 for faster training
learning_rate = 0.1
epochs = 1000  # Reduced for faster testing

# Create network
hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
output_neuron = Neuron(hidden_size)

print(f"Network: {input_size} inputs → {hidden_size} hidden → 1 output")

# Training
print("\nTraining...")
for epoch in range(epochs):
    total_error = 0
    
    for img, label in zip(train_images, train_labels):
        # Forward pass
        hidden_output = []
        for hidden_neuron in hidden_layer:
            hidden_output.append(hidden_neuron.forward(img))
        
        final_output = output_neuron.forward(hidden_output)
        
        # Error
        error = label - final_output
        total_error += error * error
        
        # Update output layer
        for i in range(len(output_neuron.weights)):
            grad = error * final_output * (1 - final_output) * hidden_output[i]
            output_neuron.weights[i] += learning_rate * grad
        
        grad_bias = error * final_output * (1 - final_output) * 1
        output_neuron.bias += learning_rate * grad_bias
        
        # Update hidden layer
        for h_idx, hidden_neuron in enumerate(hidden_layer):
        	print("doing h_indx: ",h_idx)
        	hidden_error = error * output_neuron.weights[h_idx]
        	hidden_out = hidden_output[h_idx]
            
	        for i in range(len(hidden_neuron.weights)):
	            grad = hidden_error * hidden_out * (1 - hidden_out) * img[i]
	            hidden_neuron.weights[i] += learning_rate * grad
	        
	        grad_bias = hidden_error * hidden_out * (1 - hidden_out) * 1
	        hidden_neuron.bias += learning_rate * grad_bias
if epoch % 100 == 0:
	    avg_error = total_error / len(train_images)
	    print(f"Epoch {epoch}: Avg Error = {avg_error:.6f}")

# Test
print("\n" + "="*50)
print("Testing...")
print("="*50)

correct = 0
for img, label in zip(test_images, test_labels):
    hidden_output = []
    for hidden_neuron in hidden_layer:
        hidden_output.append(hidden_neuron.forward(img))
    final_output = output_neuron.forward(hidden_output)
    
    predicted = 1 if final_output > 0.5 else 0
    if predicted == label:
        correct += 1

print(f"\nAccuracy: {correct}/{len(test_images)} = {correct/len(test_images)*100:.1f}%")

# Show some predictions
print("\n" + "="*50)
print("Sample Predictions:")
print("="*50)
for img, label in zip(test_images[:10], test_labels[:10]):
    hidden_output = []
    for hidden_neuron in hidden_layer:
        hidden_output.append(hidden_neuron.forward(img))
    final_output = output_neuron.forward(hidden_output)
    
    predicted = 1 if final_output > 0.5 else 0
    result = "✓" if predicted == label else "✗"
    print(f"  Output: {final_output:.4f} → Predicted: {'Parijat' if predicted==1 else 'Other'} | Actual: {'Parijat' if label==1 else 'Other'} {result}")