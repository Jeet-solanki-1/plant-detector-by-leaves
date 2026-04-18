

## Multi-class-classifier

Lines Changed: Binary → Multi-Class
Here are only the changed lines connected to our evolution flow.

Flow Connection Legend
Flow Step	Corresponding Change
One-Hot Encoding	Labels become [1,0,0] instead of 1
Multiple Output Neurons	output_layer = [Neuron() for _ in range(3)]
Softmax	probabilities = softmax(raw_scores)
Cross-Entropy	loss = -Σ(target × log(prediction))

prepare_data.py (lines 28-35)
    ↓
    One-Hot Encoding: [1,0,0] instead of 1
    ↓
train_model_multi.py (line 75)
    ↓
    Multiple Output Neurons (3 instead of 1)
    ↓
train_model_multi.py (line 95)
    ↓
    Softmax (probabilities sum to 1)
    ↓
train_model_multi.py (line 110)
    ↓
    Cross-Entropy Loss (punishes wrong class)
    ↓
train_model_multi.py (line 120)
    ↓
    Gradient = probabilities - target (simplified!)
    ↓
train_model_multi.py (lines 130, 140)
    ↓
    Backpropagate through all 3 outputs
    ↓
train_model_multi.py (line 200)
    ↓
    Argmax for final prediction