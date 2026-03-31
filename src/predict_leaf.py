"""
Prediction module for leaf classification.
Loads trained model and predicts on new images.
"""

import pickle
import numpy as np 
import os
import sys
from PIL import Image
from typing import Tuple

def sigmoid(x:np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1/(1+np.exp(-x))

class VectorizedNeuron:
    """Neuron for prediction (weights loaded from trained model)."""

    def __init__(self,num_inputs:int):
        self.weights=None
        self.bias=0.0

    def forward(self, inputs:np.ndarray) -> float:
        """Forward pass: z = w*x + b, output = sigmoid(z)."""
        z = np.dot(inputs, self.weights) + self.bias
        return sigmoid(z)

def prepare_image(image_path: str, target_size: Tuple[int,int]) -> np.ndarray:
    