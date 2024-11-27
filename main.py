from neuron import Neuron
from neural_network import NeuralNetwork
from mse_loss import mse_loss
import numpy as np

weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

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

# Train neural network
network.train(data, all_y_trues)

x = np.array([2, 3])
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2]) # 155 pounds, 68 inches

print(f"Neuron feedforward: {n.feedforward(x)}")
print(f"Neural Network: {network.feedforward(x)}")
print(f"Mean Squared Error: {mse_loss(y_true, y_pred)}")

# If closer to one, then is a female.
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))
