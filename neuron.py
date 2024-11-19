import numpy as np
from sigmoid import sigmoid

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed_forward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    