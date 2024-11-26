import numpy as np
from sigmoid import sigmoid
from neuron import Neuron


class NeuralNetwork:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # weights = np.array([0, 1])
        # bias = 0
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        # self.h1 = Neuron(weights, bias)
        # self.h2 = Neuron(weights, bias)
        # self.ol = Neuron(weights, bias)

    # h1 & h2 are hidden layers
    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        ol = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)



        # Output layer
        ol = self.ol.feedforward(np.array([h1, h2]))

        return ol
