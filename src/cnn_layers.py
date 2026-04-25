import numpy as np

class Conv2D:
    """
    Applies a 2D convolution to input.
    
    What you need to implement:
    - forward(): loop over each filter, each output position
    - backward(): compute gradients w.r.t filters and input
    
    Attributes:
        n_filters   : number of filters (e.g., 16)
        filter_size : e.g., 3
        stride      : step size (e.g., 1)
        padding     : zero padding size (e.g., 0 or 1)
        weights     : shape (n_filters, filter_size, filter_size, n_channels)
        bias        : shape (n_filters,)
    """
    def __init__(self, n_filters, filter_size, stride=1, padding=0):
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        # weights and bias will be initialized when input shape is known
        self.weights = None
        self.bias = None
        self.cache = None   # store values needed for backward pass
    
    def init_weights(self, input_shape):
        """He/Xavier initialization based on input shape"""
        n_channels = input_shape[0]
        # fill with random values scaled appropriately
        pass
    
    def forward(self, input):
        """
        input shape: (channels, height, width)
        output shape: (n_filters, out_height, out_width)
        """
        # 1. Add padding to input if needed
        # 2. Calculate output dimensions
        # 3. Loop over each filter
        #    Inside: loop over output height, output width
        #           compute dot product between filter and input patch
        # 4. Store input and any intermediate values in self.cache
        pass
    
    def backward(self, grad_output, learning_rate):
        """
        grad_output: gradients from next layer (shape same as forward output)
        Updates self.weights and self.bias
        Returns gradient to pass to previous layer
        """
        pass


class MaxPool2D:
    """
    Down-sampling by taking max in a 2x2 window.
    
    Attributes:
        pool_size : e.g., 2
        stride    : usually same as pool_size
    """
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, input):
        """
        input shape: (channels, height, width)
        output shape: (channels, out_height, out_width)
        """
        # For each channel, for each window position, take max value
        # Remember which position the max came from (for backward)
        pass
    
    def backward(self, grad_output, learning_rate):
        """
        Place gradients only where the max came from.
        """
        pass


class ReLU:
    """Activation: max(0, x)"""
    def forward(self, input):
        self.cache = input
        return np.maximum(0, input)
    
    def backward(self, grad_output, learning_rate):
        """grad_output is 1 where input > 0, else 0"""
        pass


class Flatten:
    """Reshape multi-dimensional input to 1D vector"""
    def forward(self, input):
        self.cache = input.shape
        return input.flatten()
    
    def backward(self, grad_output, learning_rate):
        """Reshape back to original shape"""
        pass


class Dropout:
    """Randomly zero out some neurons during training to prevent overfitting"""
    def __init__(self, rate=0.5):
        self.rate = rate          # fraction of neurons to zero
        self.mask = None          # which neurons were kept (for backward)
        self.training = True
    
    def forward(self, input):
        if self.training:
            # create random mask, scale to keep expected sum constant
            pass
        else:
            # at test time, no dropout (just pass through)
            return input
    
    def backward(self, grad_output, learning_rate):
        """Apply the same mask to the gradient"""
        pass


class Dense:
    """Fully connected layer (same as your VectorizedNeuron but for multiple outputs)"""
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_outputs, n_inputs) * np.sqrt(2.0 / n_inputs)
        self.bias = np.zeros(n_outputs)
        self.cache = None
    
    def forward(self, input):
        """input shape: (n_inputs,), output shape: (n_outputs,)"""
        pass
    
    def backward(self, grad_output, learning_rate):
        """Compute gradients and update weights"""
        pass


class SoftmaxCrossEntropy:
    """
    Combine softmax activation and cross-entropy loss for numerical stability.
    Returns both loss and gradients.
    """
    def forward(self, logits, target_one_hot):
        """
        logits: raw scores from last Dense layer (shape: (3,))
        target_one_hot: shape (3,)
        Returns: loss (scalar)
        """
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Cross-entropy loss
        loss = -np.sum(target_one_hot * np.log(probs + 1e-8))
        
        # Gradient for logits (this is the key formula)
        grad_logits = probs - target_one_hot
        
        self.cache = grad_logits
        return loss, grad_logits