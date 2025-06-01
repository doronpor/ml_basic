import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def generate_spiral_data(n_samples_per_class=100, n_classes=3, noise=0.1):
    """Generate spiral data for classification"""
    X = []
    y = []
    
    for i in range(n_classes):
        r = np.linspace(0.0, 1, n_samples_per_class)
        t = np.linspace(i * 4, (i + 1) * 4, n_samples_per_class) + np.random.randn(n_samples_per_class) * noise
        X.append(np.column_stack((r * np.sin(t * 2.5), r * np.cos(t * 2.5))))
        y.extend([i] * n_samples_per_class)
    
    return np.vstack(X), np.array(y)

def standardize(X):
    """Standardize features to have zero mean and unit variance"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8), mean, std

def train_test_split(X, y, test_size=0.2, random_seed=None):
    """Split data into training and test sets"""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create random permutation of indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # Initialize weights and biases with He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []

    def forward(self, X):
        """Forward pass"""
        # Hidden layer with ReLU
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        
        # Output layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2) # probability distribution over classes
        
        return self.A2

    def backward(self, X, y, y_pred, batch_size):
        """Backward pass"""
        # Convert y to one-hot encoding if it's not already
        if y.ndim == 1:
            y_one_hot = np.zeros((y.size, y_pred.shape[1]))
            y_one_hot[np.arange(y.size), y] = 1
        else:
            y_one_hot = y
        
        # Output layer gradients
        dZ2 = y_pred - y_one_hot
        dW2 = (1/batch_size) * self.A1.T @ dZ2
        db2 = (1/batch_size) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients with ReLU derivative
        dZ1 = np.dot(dZ2, self.W2.T) * relu_derivative(self.Z1)
        dW1 = (1/batch_size) * X.T @ dZ1
        db1 = (1/batch_size) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def fit(self, X, y, batch_size=32, epochs=100):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                y_batch_one_hot = np.zeros((y_batch.size, y_pred.shape[1]))
                y_batch_one_hot[np.arange(y_batch.size), y_batch] = 1
                batch_loss = -np.mean(y_batch_one_hot * np.log(y_pred + 1e-15))
                epoch_loss += batch_loss
                
                # Backward pass
                self.backward(X_batch, y_batch, y_pred, end_idx - start_idx)
            
            # Store average epoch loss
            self.loss_history.append(epoch_loss / n_batches)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {self.loss_history[-1]:.4f}")

    def predict(self, X):
        """Predict class labels"""
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.forward(X)

def plot_decision_boundary(model, X, y):
    """Plot the decision boundary of the model"""
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

# Example usage
if __name__ == "__main__":
    # Generate spiral dataset
    X, y = generate_spiral_data(n_samples_per_class=100, n_classes=3, noise=0.1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
    
    # Standardize features
    X_train, mean, std = standardize(X_train)
    X_test = (X_test - mean) / (std + 1e-8)
    
    # Create and train the model
    model = NeuralNetworkClassifier(input_size=2, hidden_size=10, output_size=3, learning_rate=0.1)
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot decision boundary
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, X_test, y_test)
    
    # Plot loss history
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 