import numpy as np

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))