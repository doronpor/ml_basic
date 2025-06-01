import numpy as np

class ConvLayer:
    def __init__(self, input_shape, kernel_size, num_filters, stride=1, padding=0):
        """
        Initialize convolutional layer
        input_shape: (height, width, channels)
        kernel_size: size of the convolution kernel (assumed square)
        num_filters: number of convolution filters
        """
        self.input_shape = input_shape
        self.in_h, self.in_w, self.in_c = input_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        
        # Calculate output dimensions
        self.out_h = (self.in_h - kernel_size + 2 * padding) // stride + 1
        self.out_w = (self.in_w - kernel_size + 2 * padding) // stride + 1
        
        # Initialize weights and bias with He initialization
        self.W = np.random.randn(kernel_size, kernel_size, self.in_c, num_filters) * np.sqrt(2.0 / (kernel_size * kernel_size * self.in_c))
        self.b = np.zeros((1, 1, 1, num_filters))
        
        # Cache for backward pass
        self.X_col = None
        self.W_col = None
        self.X_pad = None

    def im2col(self, X, h_out, w_out):
        """Convert input image to columns for efficient convolution"""
        X_padded = np.pad(X, ((0,0), (self.padding, self.padding), 
                             (self.padding, self.padding), (0,0)), 
                         mode='constant')
        self.X_pad = X_padded
        
        batch_size = X.shape[0]
        channels = X.shape[3]
        
        # Initialize output matrix
        X_col = np.zeros((batch_size, h_out * w_out, self.kernel_size * self.kernel_size * channels))
        
        # Convert image to columns
        for h in range(h_out):
            for w in range(w_out):
                h_start = h * self.stride
                h_end = h_start + self.kernel_size
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                
                # Extract patches and reshape
                patch = X_padded[:, h_start:h_end, w_start:w_end, :]
                X_col[:, h * w_out + w, :] = patch.reshape(batch_size, -1)
        
        return X_col

    def col2im(self, dX_col, X_shape):
        """Convert columns back to image format"""
        batch_size, h, w, c = X_shape
        h_pad = h + 2 * self.padding
        w_pad = w + 2 * self.padding
        dX_pad = np.zeros((batch_size, h_pad, w_pad, c))
        
        for h in range(self.out_h):
            for w in range(self.out_w):
                h_start = h * self.stride
                h_end = h_start + self.kernel_size
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                
                # Reshape and accumulate gradients
                patch = dX_col[:, h * self.out_w + w, :].reshape(
                    batch_size, self.kernel_size, self.kernel_size, c)
                dX_pad[:, h_start:h_end, w_start:w_end, :] += patch
        
        # Remove padding if any
        if self.padding > 0:
            dX = dX_pad[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_pad
            
        return dX

    def forward(self, X):
        """
        Forward pass of convolution
        X: Input of shape (batch_size, height, width, channels)
        Returns: Output of shape (batch_size, out_height, out_width, num_filters)
        """
        self.X = X
        batch_size = X.shape[0]
        
        # Convert image to columns
        X_col = self.im2col(X, self.out_h, self.out_w)
        self.X_col = X_col
        
        # Reshape weights
        W_col = self.W.reshape(-1, self.num_filters)
        self.W_col = W_col
        
        # Compute convolution as matrix multiplication
        out = X_col @ W_col + self.b.reshape(1, -1)  # Shape: (batch_size, h*w, num_filters)
        out = out.reshape(batch_size, self.out_h, self.out_w, self.num_filters)
            
        return out

    def backward(self, dout):
        """
        Backward pass of convolution
        dout: Gradient of loss with respect to output
        Returns: (dX, dW, db) gradients with respect to input, weights, and bias
        """
        batch_size = dout.shape[0]
        
        # Reshape dout
        dout_reshaped = dout.reshape(batch_size, -1, self.num_filters)
        
        # Gradient with respect to bias
        db = np.sum(dout_reshaped, axis=(0, 1)) / batch_size  # Average over batch
        db = db.reshape(1, 1, 1, -1)
        
        # Gradient with respect to W
        dW = np.zeros_like(self.W)
        for i in range(batch_size):
            dW_i = self.X_col[i].T @ dout_reshaped[i]
            dW += dW_i.reshape(self.kernel_size, self.kernel_size, self.in_c, self.num_filters)
        dW /= batch_size  # Average over batch
        
        # Gradient with respect to X
        W_reshape = self.W.reshape(-1, self.num_filters)
        dX_col = np.zeros_like(self.X_col)
        for i in range(batch_size):
            dX_col[i] = dout_reshaped[i] @ W_reshape.T
        
        dX = self.col2im(dX_col, self.X.shape)
        
        return dX, dW, db

# Example usage
if __name__ == "__main__":
    # Create sample input
    batch_size = 2
    height = 28
    width = 28
    channels = 1
    X = np.random.randn(batch_size, height, width, channels)
    
    # Create convolutional layer
    conv_layer = ConvLayer(
        input_shape=(height, width, channels),
        kernel_size=3,
        num_filters=16,
        stride=1,
        padding=1
    )
    
    # Forward pass
    out = conv_layer.forward(X)
    print("Output shape:", out.shape)
    
    # Backward pass (assume gradient from next layer)
    dout = np.random.randn(*out.shape)
    dX, dW, db = conv_layer.backward(dout)
    print("\nGradient shapes:")
    print("dX shape:", dX.shape)
    print("dW shape:", dW.shape)
    print("db shape:", db.shape) 