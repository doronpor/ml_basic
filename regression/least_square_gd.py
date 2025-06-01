from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class LeastSquaresGD:
    """
    Least Squares regression using gradient descent optimization.

    This class implements multivariate least squares regression for multiple targets,
    optimizing the parameters using gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size used in gradient descent optimization
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: Optional[npt.NDArray[np.float64]] = None
        self.bias: Optional[npt.NDArray[np.float64]] = None
        self.loss_history: list[float] = []

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        """
        Fit the least squares model using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples, n_targets)
            Target values

        Returns
        -------
        None
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        _, n_targets = y.shape
        self.weights = np.zeros((n_features, n_targets))
        self.bias = np.zeros((1, n_targets))

        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = self.predict(X)

            # Compute gradients
            dw = (1 / n_samples) * X.T @ (y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y, axis=0, keepdims=True)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute and store loss
            loss = np.mean((y_predicted - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make predictions for given input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data

        Returns
        -------
        array-like of shape (n_samples, n_targets)
            Predicted values

        Raises
        ------
        ValueError
            If model hasn't been fitted yet
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model must be fitted before making predictions")
        return X @ self.weights + self.bias


# Generate sample data with 3 features and 2 target variables
np.random.seed(42)
n_samples = 100
n_features = 3
n_targets = 2

# Generate X with 3 features
X = np.random.randn(n_samples, n_features)

# True parameters (weights and biases)
true_weights = np.array(
    [
        [2.0, 1.0],  # weights for feature 1
        [-1.0, 0.5],  # weights for feature 2
        [3.0, -1.0],  # weights for feature 3
    ]
)
true_bias = np.array([[4.0, 2.0]])

# Generate y with 2 target variables using the true parameters
y = X @ true_weights + true_bias + np.random.randn(n_samples, n_targets) * 0.1

# Create and train the model
model = LeastSquaresGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions on training data
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(15, 5))

# Plot actual vs predicted for first target variable
plt.subplot(1, 3, 1)
plt.scatter(y[:, 0], y_pred[:, 0], color="blue", alpha=0.5)
plt.plot([y[:, 0].min(), y[:, 0].max()], [y[:, 0].min(), y[:, 0].max()], "r--", lw=2)
plt.xlabel("Actual Values (Target 1)")
plt.ylabel("Predicted Values (Target 1)")
plt.title("Actual vs Predicted - Target 1")

# Plot actual vs predicted for second target variable
plt.subplot(1, 3, 2)
plt.scatter(y[:, 1], y_pred[:, 1], color="green", alpha=0.5)
plt.plot([y[:, 1].min(), y[:, 1].max()], [y[:, 1].min(), y[:, 1].max()], "r--", lw=2)
plt.xlabel("Actual Values (Target 2)")
plt.ylabel("Predicted Values (Target 2)")
plt.title("Actual vs Predicted - Target 2")

# Plot the loss history
plt.subplot(1, 3, 3)
plt.plot(model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Loss History")

plt.tight_layout()
plt.show()

# Print the learned parameters
print("True weights:")
print(true_weights)
print("\nLearned weights:")
print(model.weights)
print("\nTrue bias:")
print(true_bias)
print("\nLearned bias:")
print(model.bias)
