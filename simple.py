import numpy as np
import random

class SimplePerceptron(object):

    def __init__(self, n_iterations=10, eta=0.01):
        self.n_iterations = n_iterations
        self.eta = eta

    def train(self, train_in, y):
        ones = np.ones((train_in.shape[0], 1))
        train_in = np.concatenate((ones, train_in), axis=1)
        i = 0
        error_min = np.inf
        error = 1
        self.w = np.zeros(len(train_in[0]))
        w_min = self.w
        while error > 0 and i < self.n_iterations:
            x_i = random.randint(0,len(train_in)-1)
            predicted = self.eta * (y[x_i] - self.predict(train_in[x_i]))
            self.w += predicted * train_in[x_i]
            # self.w[0] += predicted
            error = 0
            for xi, yi in zip(train_in, y):
                error += (yi-self.predict(xi))**2
            error/=2
            if error < error_min:
                error_min = error
                w_min = self.w
            i += 1
            # print(error, error_min)
        if i >= self.n_iterations:
            print("Error: Conjunto inseparable")
            exit(1)

        return self

    def activation(self, weighted_sum):
        return np.where(weighted_sum >= 0.0, 1, 0)
    
    def predict(self, input):
        return self.activation(np.dot(input, self.w)) 

    def prediction(self, input):
        ones = np.ones((input.shape[0], 1))
        input = np.concatenate((ones, input), axis=1)
        return self.activation(np.dot(input, self.w))

    def get_error(self, err):
        return err


# inputs = [[1,1], [1,0], [0,1], [0,0]]
# outputs = np.array([1, 0, 0, 0])

# perceptron = SimplePerceptron()
# perceptron.train(np.array(inputs), outputs)

# inputs = np.array([[1,1], [1,0], [1,1]])
# print(perceptron.predict(inputs)) 