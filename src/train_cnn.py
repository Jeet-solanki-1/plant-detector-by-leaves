"""
CNN training script for leaf classification.
Uses the same prepare_data.py as before.
"""

from prepare_data import load_data
from cnn_layers import Conv2D, MaxPool2D, ReLU, Flatten, Dense, Dropout, SoftmaxCrossEntropy
import numpy as np
import time
import pickle
import os

class CNNClassifier:
    def __init__(self, input_shape=(3, 64, 64), num_classes=3):
        """
        input_shape: (channels, height, width) e.g., (3, 64, 64)
        """
        self.layers = []
        self._build_architecture(input_shape)
        self.loss_fn = SoftmaxCrossEntropy()
        self.training_history = {'epoch': [], 'loss': [], 'accuracy': []}
    
    def _build_architecture(self, input_shape):
        """
        Define the sequence of layers.
        This is where you design the CNN.
        """
        # Example architecture (you can change numbers)
        current_shape = input_shape
        
        # Conv1 → ReLU → MaxPool
        self.layers.append(Conv2D(n_filters=16, filter_size=3, stride=1, padding=1))
        # Need to initialize weights after knowing input shape
        self.layers[-1].init_weights(current_shape)
        # Compute new shape after conv (not implemented in skeleton)
        
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(pool_size=2, stride=2))
        # Update current_shape after pooling
        
        # Conv2 → ReLU → MaxPool
        # self.layers.append(Conv2D(...))
        # ...
        
        # Flatten → Dense → Dropout → Dense
        self.layers.append(Flatten())
        self.layers.append(Dense(n_inputs=???, n_outputs=128))
        self.layers.append(Dropout(rate=0.5))
        self.layers.append(Dense(n_inputs=128, n_outputs=3))
    
    def forward(self, input):
        """Pass input through all layers"""
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits (not probabilities)
    
    def backward(self, grad_logits, learning_rate):
        """Pass gradient backward through all layers (reverse order)"""
        grad = grad_logits
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad
    
    def train_step(self, image, target_one_hot, learning_rate):
        # Forward pass
        logits = self.forward(image)
        
        # Compute loss and gradient w.r.t logits
        loss, grad_logits = self.loss_fn.forward(logits, target_one_hot)
        
        # Backward pass
        self.backward(grad_logits, learning_rate)
        
        return loss
    
    def train(self, train_images, train_labels, batch_size=16, learning_rate=0.001, epochs=50):
        """Main training loop (similar to your MLP train method)"""
        n_samples = len(train_images)
        # ... implement shuffling, batching, checkpointing, early stopping
        # Same structure as your MLP, but calling train_step instead of train_batch
        pass
    
    def predict(self, image):
        """Return probabilities for a single image (after softmax)"""
        logits = self.forward(image)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def evaluate(self, test_images, test_labels):
        """Same as MLP evaluate"""
        pass
    
    def save(self, path):
        """Save all layer parameters"""
        pass


def main():
    # Load data (same as before)
    images, labels = load_data("../data/healthy", target_size=(64,64), ...)
    # Preprocess: reshape to (channels, height, width) = (3, 64, 64)
    
    # Split train/test
    
    # Create model
    model = CNNClassifier(input_shape=(3, 64, 64))
    
    # Train
    model.train(train_images, train_labels, batch_size=16, epochs=50)
    
    # Evaluate
    acc, preds = model.evaluate(test_images, test_labels)
    print(f"Test Accuracy: {acc:.1f}%")
    
    # Save model
    model.save("../traind_models/cnn_leaf_classifier.pkl")

if __name__ == "__main__":
    main()