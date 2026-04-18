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
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation for multi-class classification.
    Converts raw scores to probabilities taht sum to 1.

    Example: [2.5, 1.2, 0.3] -> [0.75, 0.18, 0.07]\
    """

    #Subtract max for numerical statbility
    exp_x = np.exp(x-np.max(x))
    return exp_x/np.sum(exp_x)


# Neuron Class

class VectorizedNeuron:
    """Neuron with vectorized forward pass using NumPy."""

    def __init__(self, num_inputs:int):
        self.weights = np.random.randn(num_inputs) * np.sqrt(1.0 / num_inputs)  # Xavier
        self.bias = 0.0

    def forward(self,inputs:np.ndarray) -> float:
        """Forward pass: z = w*x +b, output = sigmoid(z)."""
        z = np.dot(inputs, self.weights) + self.bias
        return sigmoid(z)

    def update_weights(self, grad_weights:np.ndarray,grad_bias: float, lr: float) -> None:
        self.weights-=lr * grad_weights
        self.bias-= lr * grad_bias

# Mulit-class Neural Network Class

class LeafClassifier:
    """
    2-Layer Neural Network for MULTI-CLASS classification.

    Architecture:
        Input layer: num_inputs neurons
        Hidden layer: hidden_size neurons (sigmoid)
        Output layer: 3 neuron (Soft_Max)
    Output: Probabilities for each class that sum to 1
    
    Training features:
        - Checkpoint saving (every N epochs)
        - Early stopping
        - Learning rate scheduling
        - Emergency save on interrupt
    """
    NUM_CLASSES=3 
    CLASS_NAME =["Parijat","Mango","Money-plant"]
    def __init__(self, input_size:int, hidden_size:int):
        """Initialize network with random weights."""
        self.hidden_layer = [VectorizedNeuron(input_size) for _ in range(hidden_size)]
        self.output_layer =[VectorizedNeuron(hidden_size) for _ in range(self.NUM_CLASSES)]
        self.training_history: Dict[str, List] = {'epoch': [], 'loss': [], 'accuracy': []}
        self._register_signal_handler()
        self._current_epoch = 0

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
            'output_weights': [n.weights.copy() for n in self.output_layer],
            'output_bias': [n.bias for n in self.output_layer],
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
        for i, out in enumerate(self.output_layer):
            out.weights = checkpoint['output_weights'][i]
            out.bias = checkpoint['output_bias'][i]
        return checkpoint.get('epoch',0)

    def forward(self,inputs: np.ndarray) -> Tuple[float,List[float]]:
        """
        Forward pass: returns  (output_probabilities, hidden_output)
        output_probabilities shape: (num_classes,) - sums to 1    
        """
        #Hidden layer (same as before)
        hidden_outputs = [n.forward(inputs) for n in self.hidden_layer]
        
        # ✅ Raw dot product only — no sigmoid on output layer
        raw_scores = np.array([
            np.dot(hidden_outputs, n.weights) + n.bias 
            for n in self.output_layer
        ])
        
        probabilities = softmax(raw_scores)
        return probabilities, hidden_outputs
       

    def train_batch(self, img: np.ndarray, label: np.ndarray, lr: float):
        """
            Train on single image WITHOUT updating weights with MULTI-CALSS target.
            Return (cross-entropy loss, hidden_out, hidden_grad, output_grad)

        """

        #Forward pass
        probabilities, hidden_out = self.forward(img)

        #Error- CHANGED: Cross-entropy loss (instead of MAE)
        # Add small epsilon to avoid log(0) but what happens if log(0) comes??

        epsilon = 1e-8
        loss = -np.sum(label*np.log(probabilities+epsilon))
        
        # CHANGED: Gradient for output layer
        # for cross-entropy + softmax, gradient = probabilities - targe(label)
        
        output_gradients = probabilities-label #shape: (num_classes,)

        #Store gradients  (NOT update yet!) update after the batch!
       


        hidden_grads=[]

        for h_idx, neuron in enumerate(self.hidden_layer):
            hidden_error = 0
            
            for i in range(self.NUM_CLASSES):
                hidden_error+= output_gradients[i]* self.output_layer[i].weights[h_idx]
            
            hidden_out_val = hidden_out[h_idx]
            hidden_grad = hidden_error*hidden_out_val*(1-hidden_out_val)
            
            hidden_grads.append({
                'grad_weights':hidden_grad*img,
                'grad_bias': hidden_grad,
                'neuron_idx': h_idx
                })

        return loss, hidden_out, output_gradients, hidden_grads

    def train(
        self, 
        train_images: np.ndarray,
        train_labels: np.ndarray,
        batch_size=32,
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

        n_samples = len(train_images)
        n_batches = (n_samples+batch_size-1)//batch_size

        best_loss= float('inf')
        current_lr=learning_rate
        patience_counter=0
        start_time= time.time()

        for epoch in range(epochs):
          
            self._current_epoch = epoch
            total_loss=0

            for batch in range(n_batches):
                start=batch*batch_size
                end = min(n_samples, start+batch_size)

                batch_imgs = train_images[start:end]
                batch_labels = train_labels[start:end]
                batch_loss=0

                # accumulated vars
                #CHANGED! NOW LOOPS AS HIDDEN LAYER AS OUTPUT IS ALSO A LAYER!
                
                accumulated_output_weights_grad= [np.zeros_like(n.weights) for n in self.output_layer]
                accumulated_output_bias_grad = [0.0 for _ in range(self.NUM_CLASSES)]
                accumulated_hidden_bias_grad = [0.0 for _ in range(len(self.hidden_layer))]
                accumulated_hidden_weights_grad = [np.zeros_like(n.weights) for n in self.hidden_layer]

                # now training 
                for img, label in zip(batch_imgs,batch_labels):
                    loss, hidden_out, output_gradients, hidden_grads = self.train_batch(img,label,current_lr)
                    batch_loss+=loss

                    
                    #CHANGED! now loops over neurons in output layer and for each accumulate!
                    for i in range(self.NUM_CLASSES):
                        accumulated_output_weights_grad[i]+=output_gradients[i]*np.array(hidden_out)
                        accumulated_output_bias_grad[i]+=output_gradients[i]

                    #build accumulated hiddden grads and bias by looping on list of them
                
                    
                    for entry in hidden_grads:
                        accumulated_hidden_bias_grad[entry['neuron_idx']]+=entry['grad_bias']
                        accumulated_hidden_weights_grad[entry['neuron_idx']]+=entry['grad_weights']
                
                batch_actual_size = len(batch_imgs)
                
                # now update output weights
                for i in range(self.NUM_CLASSES):
                    self.output_layer[i].update_weights(
                        accumulated_output_weights_grad[i]/batch_actual_size,
                        accumulated_output_bias_grad[i]/batch_actual_size,
                        current_lr
                        )

                # now update hidden weights
                for h_idx  in range(len(self.hidden_layer)):
                    self.hidden_layer[h_idx].update_weights(
                        accumulated_hidden_weights_grad[h_idx]/batch_actual_size,
                        accumulated_hidden_bias_grad[h_idx]/batch_actual_size,
                        current_lr
                        )
                # updated weights

                total_loss+=batch_loss



            avg_loss = total_loss/n_samples

            # Calculate training accuracy
            #UPDATED-FOR-MULTI-CLASS: AS now it uses argmax and then find the ans whoever has the max number winss!

            train_correct = 0
            for img, label in zip(train_images, train_labels):
                probs, _ = self.forward(img)
                predicted = np.argmax(probs)
                actual = np.argmax(label)
                if predicted == actual:
                    train_correct+=1
            train_acc = train_correct / len(train_images)*100

            # Store history
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(train_acc)

            # Print progress (minimal)
            if epoch % 1==0:
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

    # Evaluate 
    def evaluate(self, test_images:np.ndarray, test_labels: np.ndarray) -> Tuple[float,np.ndarray]:
        predictions = []
        correct=0
        for img,label in zip(test_images,test_labels):

            probs,_ = self.forward(img)
            predicted = np.argmax(probs)
            actual = np.argmax(label)

            predictions.append(predicted)
            if predicted==actual:
                correct+=1
        return correct/len(test_images) * 100 , np.array(predictions)

    #TODO: self rewrite
    def save(self, filename: str = "leaf_model.pkl") -> None:
        """Save full model (weights + training history)."""
        
        
        model_data={
            'hidden_weights':[neuron.weights for neuron in self.hidden_layer],
            'hidden_biases':[neuron.bias for neuron in self.hidden_layer],
            'output_weights':[n.weights for n in self.output_layer],
            'output_bias': [n.bias for n in self.output_layer],
            'training_history':self.training_history
        }

        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name=filename.replace('.pkl','') if filename.endswith('.pkl') else filename
        save_path=f"{base_name}_{timestamp}.pkl"
        with open(save_path,'wb') as f:
            pickle.dump(model_data,f)

# Main Training Script

def main():
    from prepare_data import load_data
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "traind_models"), exist_ok=True)
    print("=" * 50)
    print("LEAF CLASSIFIER TRAINING")
    print("=" * 50)

    # Load data
    images, labels = load_data(
        "../data/healthy",
        target_size=(98,98),
        verbose=True,
        augment=True,
        augment_factor=3,
        augment_intensity="medium",
        max_per_class=19  # 19 originals × 4 (1 + augment_factor=3) = 76 per class
        )

    print(f"Dataset size: {len(images)} images (with augmentation)")
    flattened = images.reshape(images.shape[0],-1)

    #Shuffle and split
    indices = np.random.permutation(len(flattened))
    flattened = flattened[indices]

    labels = labels[indices]
     # Add this in main() after loading data, before model.train()
    print("\n🔍 DATA DIAGNOSTIC:")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"First 5 labels:\n{labels[:5]}")
    print(f"Argmax of first 5 labels: {[np.argmax(labels[i]) for i in range(5)]}")
    print(f"Label sum (should equal num_samples): {np.sum(labels)}")

    # Check class distribution
    class_counts = [np.sum(labels[:, i]) for i in range(3)]
    print(f"Class counts: Parijat={class_counts[0]}, Mango={class_counts[1]}, Money-plant={class_counts[2]}")
    split = int(0.8*len(flattened))
    train_images = flattened[:split]
    train_labels = labels[:split]
    test_images = flattened[split:]
    test_labels = labels[split:]

    # Create model 
    input_size = 98*98*3
    hidden_size = 50

    model = LeafClassifier(input_size,hidden_size)

    print(f"Input: {input_size}, Hidden: {hidden_size}, Parameters: {input_size*hidden_size + hidden_size + hidden_size + 1:,}")

    # Train
    model.train(train_images, train_labels,batch_size=32, learning_rate=0.01, epochs=50,checkpoint_interval=50,checkpoint_dir="../checkpoints")

    # Evaluate
    acc, preds = model.evaluate(test_images, test_labels)
    print(f"\nTest Accuracy: {acc:.1f}% ")

    # Save final
    model.save("../traind_models/leaf_model_multi_class_with_Xavier_init_of_weights.pkl")
    print("Model saved.")

if __name__=="__main__":
    main()
