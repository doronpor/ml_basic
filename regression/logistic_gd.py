import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the logistic regression model using gradient descent

        Parameters:
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_pred = X @ self.weights + self.bias
            predictions = self.sigmoid(linear_pred)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.loss_history.append(loss)

            # Compute gradients
            dz = predictions - y
            dw = (1 / n_samples) * X.T @ dz
            db = (1 / n_samples) * np.sum(dz)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Predict probability of class 1"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        """Predict class labels (0 or 1)"""
        return (self.predict_proba(X) >= threshold).astype(int)


def generate_binary_classification_data(n_samples=100, noise=0.1):
    """Generate synthetic data for binary classification"""
    # Generate two gaussian clouds
    n_samples_per_class = n_samples // 2

    # Class 0
    X0 = np.random.randn(n_samples_per_class, 2)
    X0 = X0 + np.array([-2, -2])  # Shift mean

    # Class 1
    X1 = np.random.randn(n_samples_per_class, 2)
    X1 = X1 + np.array([2, 2])  # Shift mean

    # Combine data
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

    # Add noise
    X += np.random.randn(*X.shape) * noise

    return X, y


def plot_decision_boundary(model, X, y):
    """Plot the decision boundary"""
    # Set min and max values for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Make predictions for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")


if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_binary_classification_data(n_samples=200, noise=0.2)

    # Create and train model
    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(15, 5))

    # Plot decision boundary
    plt.subplot(1, 2, 1)
    plot_decision_boundary(model, X, y)
    plt.title("Decision Boundary")

    # Plot loss history
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
