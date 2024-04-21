import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def get_activation_function(name: str):
    if name == "sigmoid":
        return sigmoid, sigmoid_derivative
    elif name == "relu":
        return relu, relu_derivative
    elif name == "tanh":
        return np.tanh, tanh_derivative
    else:
        raise ValueError(f"Unknown activation function: {name}")


def max_pooling(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    return np.array(
        [
            np.max(x[i : i + pool_size, j : j + pool_size])
            for i in range(0, x.shape[0], pool_size)
            for j in range(0, x.shape[1], pool_size)
        ]
    ).reshape(x.shape[0] // pool_size, x.shape[1] // pool_size)
