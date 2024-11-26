import numpy as np
from sigmoid import sigmoid, deriv_sigmoid
from neuron import Neuron
from mse_loss import mse_loss

class NeuralNetwork:
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    # h1 & h2 are hidden layers
    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        ol = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3) # Output layer

        return ol

    # Train
    # data is (n x 2) numpy array, where n = no. of samples in dataset.
    # all_y_trues is numpy array with n elements.
    # Elements in all_y_trues correspond to those in data.
    def train(self, data, all_y_trues):
        learn_rate = 0.1

        # Number of counts to loop entire .
        epochs = 1000 
    
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_ol = self.w5 * h1 + self.w6 * h2 + self.b3
                ol = sigmoid(sum_ol)
                y_pred = ol

                # Calculate partial derivatives
                d_l_d_ypred = -2 * (y_true - y_pred)

                # Neuron ol
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_ol)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_ol)
                d_ypred_d_b3 = deriv_sigmoid(sum_ol)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_ol)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_ol)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                # Nueron h1
                self.w1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron ol
                self.w5 -= learn_rate * d_l_d_ypred * d_ypred_d_w5 
                self.w6 -= learn_rate * d_l_d_ypred * d_ypred_d_w6 
                self.b3 -= learn_rate * d_l_d_ypred * d_ypred_d_b3 

            # Calculate total loss at the end of each epoch.
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
         
