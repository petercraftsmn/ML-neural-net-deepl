import numpy as np


class Network(object):
    """
    Represents neural network class.
    """

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a


def sigmoid(z):
    """Sigmoid function

    When the input z is a vector or Numpy array,
    Numpy automatically applies the function sigmoid elementwise,
    that is, in vectorized form
    """
    return 1.0 / (1.0 + np.exp(-z))
