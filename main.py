from neuron import Neuron
import numpy as np

weights = np.array([0, 1]) 
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
print(n.feed_forward(x))