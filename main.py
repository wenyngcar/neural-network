from neuron import Neuron
from neural_network import NeuralNetwork
import numpy as np

weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
network = NeuralNetwork()

print(n.feedforward(x))
print(network.feedforward(x))
