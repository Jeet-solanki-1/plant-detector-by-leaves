"""
Neural network module for leaf clasification.
Implements a 2-layer netowrk with sigmoid activation.
"""

import numpy as np
import pickle
import signal
import sys
import time
from typing import Tuple, List, Optional, Dict
from datetime import datetime


# Activation Functions

def sigmoid(x:np.ndarray) -> np.ndarray:
    """Vectorized sigmoid activation."""
    return 1/(1+np.exp(-x))

# Neuron Class

class VectorizedNeuron:
    """Neuron with vectorized forward pass using NumPy."""

    def __init__(self, num_inputs:int):
        """Initialize weights uniformly in [-0.5,0.5], bias = 0."""
        self.weights = np.random.uniform(-0.5,0.5, num_inputs)
        self.bias = 0.0

    def forward(self,inputs:np.ndarray) -> float:
        """Forward pass: z = w*x +b, output = sigmoid(z)."""
        z = np.dot(inputs, self.weights) + self.bias
        return sigmoid(z)

    def update_weights(self, grad_weights:np.ndarray,grad_bias: float, lr: float) -> None:
        self.weights+=lr * grad_weights
        self.bias+= lr * grad_bias

# Neural Network Class

class LeafClassifier:
    """
    2-Layer Neural Network for binary classification.

    Architecture:
        Input layer: num_inputs neurons
        Hidden layer: hidden_size neurons (sigmoid)
        Output layer: 1 neuron (sigmoid)
    
    Training features:
        - Checkpoint saving (every N epochs)
        - Early stopping
        - Learning rate scheduling
        - Emergency save on interrupt
    """

    def __init__(self, input_size:int, hidden_size:int):
        """Initialize network with random weights."""
        self.hidden_layer = [VectorizedNeuron(input_size) for _ in range(hidden_size)]
        self.output_neuron = VectorizedNeuron(hidden_size)
        self.training_history: Dict[str, List] = {'epoch': [], 'loss': [], 'accuracy': []}
        self._register_signal_handler()

    def _register_signal_handler(self) -> None:
        """ Register handler for Ctrl+C to save emergency checkpoint."""
        def handler(sig,frame):
            self._save_checkpoint(f"emergency_epoch_{self._current_epoch}.pkl")
            sys.exit(0)
        signal.signal(signal.SIGINT,handler)

    def _save_checkpoint(self, filename:str, epoch: int = None, loss: float = None) -> None:
        """Save current model state as checkpoint."""
        checkpoint = {
            'hidden_weights': [n.weights.copy() for n in self.hidden_layer],
            'hidden_biases': [n.bias for n in self.hidden_layer],
            'output_weights': self.output_neuron.weights.copy(),
            'output_bias': self.output_neuron.bias,
            'epoch': epoch or getattr(self, '_current_epoch',0),
            'loss': loss
        }
        with open(filename,'wb') as f:
            pickle.dump(checkpoint,f)

    def load_checkpoint(self, filename: str) -> int:
        """Load model from checkpoint file. Returns last epoch."""
        with open(filename,'rb') as f:
            checkpoint = pickle.load(f)

        for i, neuron in enumerate(self.hidden_layer):
            neuron.weights = checkpoint['hidden_weights'][i]
            neuron.bias = checkpoint['hidden_biases'][i]
        self.output_neuron.weights = checkpoint['output_weights']
        self.output_neuron.bias = checkpoint['output_bias']
        return checkpoint.get('epoch',0)

    def forward(self,inputs: np.ndarray) -> Tuple[float,List[float]]:
        """Forward pass: returns  (output, hidden_output)."""
        hidden_outputs = [n.forward(inputs) for n in self.hidden_layer]
        output = self.output_neuron.forward(np.array(hidden_outputs))
        return output, hidden_outputs

    def train_batch(self, img: np.ndarray, label: int, lr: float) -> float:
        """Train on single image. Returns squared error."""
        #Forward pass
        final_output, hidden_out = self.forward(img)

        error = label - final_output
        output_grad = error * final_output * (1-final_output)

        # Update output layer
        self.output_neuron.update_weights(
            output_grad * np.array(hidden_out),
            output_grad,
            lr
            )

        # Update hidden layer
        for h_indx, neuron in enumerate(self.hidden_layer):
            hidden_error = error * self.output_neuron.weights[h_indx]
            hidden_out_val = hidden_out[h_indx]
            hidden_grad = hidden_error * hidden_out_val * (1-hidden_out_val)
            neuron.update_weights(hidden_grad*img,hidden_grad,lr)

        return error ** 2

    def train(
        self, 
        train_images: np.ndarray,
        train_labels: np.ndarray,
        learning_rate: float = 0.03,
        epochs: int = 300,
        early_stopping_patience: int =30,
        checkpoint_interval: int = 100,
        checkpoint_dir: str = "checkpoints"
        ) -> Dict:
        """
        Train the network.

        Args:
            train_images: Training images (N, input_size)
            train_labels: Training labels (N,)
            test_images: Optional test images for validation
            test_labels: Optional test labels
            learning_rate: Initial learning rate
            epochs: Maximum epochs
            early_stopping_patience: Stop if no improvement for N epochs
            checkpoint_interval: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dict
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_loss = float('inf')
        patience_counter = 0
        current_lr = learning_rate
        start_time = time.time()

        for epoch in range(epochs):
            self._current_epoch = epoch

            # Learning rate scheduling
            if epoch == 100:
                current_lr = learning_rate/2

            # Training
            total_loss = 0.0
            for img, label in zip(train_images,train_labels):
                total_loss+=self.train_batch(img,label,current_lr)
            avg_loss = total_loss/len(train_images)

            # Calculate training accuracy
            train_correct = 0
            for img, label in zip(train_images, train_labels):
                output, _ = self.forward(img)
                if (output> 0.5) == label:
                    train_correct+=1
            train_acc = train_correct / len(train_images)*100

            # Store history
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(train_acc)

            # Print progress (minimal)
            if epoch % checkpoint_interval==0:
                print(f"Epoch {epoch:4d} | Loss {avg_loss:.6f} | Acc: {train_acc:.1f}%")

            # Save checkpoint
            if epoch > 0 and epoch % checkpoint_interval ==0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_checkpoint(
                    f"{checkpoint_dir}/{timestamp}_checkpoint_epoch_{epoch}.pkl",
                    epoch,
                    avg_loss
                    )

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter=0
                self._save_checkpoint(
                    f"{checkpoint_dir}/best_model.pkl",
                    epoch, avg_loss
                    )
            else:
                patience_counter+=1
                if patience_counter>=early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Training complete in {(time.time()-start_time)/60:.1f} min")
        return self.training_history

    #TODO: self rewrite
    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate model on test data. Returns (accuracy, predictions)."""
        predictions = []
        correct = 0
        
        for img, label in zip(test_images, test_labels):
            output, _ = self.forward(img)
            pred = 1 if output > 0.5 else 0
            predictions.append(pred)
            if pred == label:
                correct += 1
        
        return correct / len(test_images) * 100, np.array(predictions)


    #TODO: self rewrite
    def save(self, filename: str = "leaf_model.pkl") -> None:
        """Save full model (weights + training history)."""
        import os
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        
        model_data = {
            'hidden_weights': [n.weights for n in self.hidden_layer],
            'hidden_biases': [n.bias for n in self.hidden_layer],
            'output_weights': self.output_neuron.weights,
            'output_bias': self.output_neuron.bias,
            'training_history': self.training_history
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.replace('.pkl', '')
        save_path = f"{base_name}_{timestamp}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

# Main Training Script

def main():
    from prepare_data import load_data
    import os
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("traind_models", exist_ok=True)
    print("=" * 50)
    print("LEAF CLASSIFIER TRAINING")
    print("=" * 50)

    # Load data
    images, labels = load_data("data/healthy",target_size=(64,64))
    flattened = images.reshape(images.shape[0],-1)

    #Shuffle and split
    indices = np.random.permutation(len(flattened))
    flattened = flattened[indices]
    labels = labels[indices]

    split = int(0.8*len(flattened))
    train_images = flattened[:split]
    train_labels = labels[:split]
    test_images = flattened[split:]
    test_labels = labels[split:]

    # Create model 
    input_size = 64*64*3
    hidden_size = 70
    model = LeafClassifier(input_size,hidden_size)

    print(f"Input: {input_size}, Hidden: {hidden_size}, Parameters: {input_size*hidden_size + hidden_size + hidden_size + 1:,}")

    # Train
    model.train(train_images, train_labels, learning_rate=0.03, epochs=300,checkpoint_interval=1,checkpoint_dir="checkpoints")

    # Evaluate
    acc, preds = model.evaluate(test_images, test_labels)
    print(f"\nTest Accuracy: {acc:.1f}% ({sum(preds==test_labels)}/{len(test_labels)})")

    # Save final
    model.save("traind_models/leaf_model_refactored_v1_rgb_64i_70h.pkl")
    print("Model saved.")

if __name__=="__main__":
    main()