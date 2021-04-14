import numpy as np
import random

class SimplePerceptron(object):

    def __init__(self, n_iterations=10, eta=0.01):
        self.n_iterations = n_iterations
        self.eta = eta

    def train(self, train_in, y):
        i = 0
        error = 1
        total_err = 0
        self.w = np.zeros(len(train_in[0]) + 1)
        predictions = []
        # error_min = np.inf
        while error > 0 and i < self.n_iterations:
            error = 0
            for xi, expected in zip(train_in, y):
                predicted = self.eta * (expected - self.predict(xi))
                self.w[1:] += predicted * xi
                self.w[0] += predicted
                error += expected - predicted
                total_err += error
            predictions.append(predicted)
            i+=1
        return self

    def activation_function(self, weighted_sum):
        return np.where(weighted_sum >= 0.0, 1, 0)
    
    def predict(self, input):
        weighted_sum = np.dot(input, self.w[1:]) + self.w[0]
        return self.activation_function(weighted_sum) 


# inputs = [[1,1], [1,0], [0,1], [0,0]]
# outputs = np.array([1, 0, 0, 0])

# perceptron = SimplePerceptron()
# perceptron.train(np.array(inputs), outputs)

# inputs = np.array([[1,1], [1,0], [1,1]])
# print(perceptron.predict(inputs)) 