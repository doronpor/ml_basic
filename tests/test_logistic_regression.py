import numpy as np
import pytest

from regression.logistic_gd import (
    LogisticRegressionGD,
    generate_binary_classification_data,
)


def test_logistic_regression_initialization():
    """Test the initialization of LogisticRegression model"""
    model = LogisticRegressionGD(learning_rate=0.01, n_iterations=100)
    assert model.learning_rate == 0.01
    assert model.n_iterations == 100
    assert model.weights is None
    assert model.bias is None
    assert len(model.loss_history) == 0


def test_sigmoid_function():
    """Test the sigmoid activation function"""
    model = LogisticRegressionGD()

    # Test sigmoid with zero
    assert model.sigmoid(0) == 0.5

    # Test sigmoid with large positive number
    assert model.sigmoid(100) > 0.99

    # Test sigmoid with large negative number
    assert model.sigmoid(-100) < 0.01

    # Test sigmoid with array
    x = np.array([-1, 0, 1])
    result = model.sigmoid(x)
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert np.all((result >= 0) & (result <= 1))


def test_prediction_shape():
    """Test the shape of predictions"""
    model = LogisticRegressionGD()
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    y = np.random.randint(0, 2, 100)  # Binary labels

    model.fit(X, y)

    # Test predict_proba
    proba = model.predict_proba(X)
    assert proba.shape == (100,)
    assert np.all((proba >= 0) & (proba <= 1))

    # Test predict
    predictions = model.predict(X)
    assert predictions.shape == (100,)
    assert np.all((predictions == 0) | (predictions == 1))


def test_model_training():
    """Test if model can fit simple data and converge"""
    # Generate simple separable data
    X = np.array(
        [[1, 1], [2, 2], [2, 1], [3, 3], [-1, -1], [-2, -2], [-2, -1], [-3, -3]]
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    # Check if loss decreases
    assert model.loss_history[-1] < model.loss_history[0]

    # Check predictions on training data
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    assert (
        accuracy > 0.75
    )  # Should achieve at least 75% accuracy on this simple dataset


def test_generate_binary_classification_data():
    """Test the data generation function"""
    n_samples = 200
    X, y = generate_binary_classification_data(n_samples=n_samples, noise=0.1)

    # Check shapes
    assert X.shape == (n_samples, 2)
    assert y.shape == (n_samples,)

    # Check labels
    assert set(np.unique(y)) == {0, 1}

    # Check class balance
    n_zeros = np.sum(y == 0)
    n_ones = np.sum(y == 1)
    assert abs(n_zeros - n_ones) <= 1  # Should be balanced


def test_error_handling():
    """Test error handling in the model"""
    model = LogisticRegressionGD()

    # Test fitting with invalid data
    with pytest.raises(Exception):
        model.fit(np.array([[1, 2], [3, 4]]), np.array([1]))  # Mismatched dimensions

    # Test prediction without fitting
    model = LogisticRegressionGD()
    with pytest.raises(Exception):
        model.predict(np.array([[1, 2]]))
