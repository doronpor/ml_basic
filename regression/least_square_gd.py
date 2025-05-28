import numpy as np
import matplotlib.pyplot as plt

class LeastSquaresGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))  # Changed to column vector
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store loss
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) * 0.1

# Create and train the model
model = LeastSquaresGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
X_test = np.array([[0], [2]])
y_pred = model.predict(X_test)

# Plotting
plt.figure(figsize=(12, 5))

# Plot the data and regression line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Least Squares Regression')
plt.legend()

# Plot the loss history
plt.subplot(1, 2, 2)
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Loss History')

plt.tight_layout()
plt.show()

# Print the learned parameters
print(f"Learned weights: {model.weights.flatten()}")
print(f"Learned bias: {model.bias}")
