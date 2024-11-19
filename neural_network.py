import numpy as np
from neuron import Neuron


class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.ol = Neuron(weights, bias)

    # h1 & h2 are hidden layers
    def feedforward(self, x):
        h1 = self.h1.feedforward(x)
        h2 = self.h2.feedforward(x)

        # Output layer
        ol = self.ol.feedforward(np.array([h1, h2]))

        return ol
