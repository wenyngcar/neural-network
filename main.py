from neuron import Neuron
from neural_network import NeuralNetwork
from mse_loss import mse_loss
import numpy as np

weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])
network = NeuralNetwork()
y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

# Define dataset
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6]
])

all_y_trues = np.array([1, 0, 0, 1])

network.train(data, all_y_trues)

print(f"Neuron feedforward: {n.feedforward(x)}")
print(f"Neural Network: {network.feedforward(x)}")
print(f"Mean Squared Error: {mse_loss(y_true, y_pred)}")