import numpy as np
import random

class SimplePerceptron(object):

    def __init__(self, n_iterations=10, eta=0.01, linear=False):
        self.n_iterations = n_iterations
        self.eta = eta
        self.linear = linear

    def train(self, train_in, y):
        i = 0
        n=0
        error_min = np.inf
        error = 1
        self.w = np.zeros(len(train_in[0]))
        self.w = np.random.uniform(-100,100, size=(1,len(train_in[0])))[0]
        w_min = []
        while error > 0 and i < self.n_iterations:
            if n > self.n_iterations/10:
                n = 0
                self.w = np.random.uniform(-100,100, size=(1,len(train_in[0])))[0]
            x_i = random.randint(0,len(train_in)-1)
            predicted = self.eta * (y[x_i] - self.predict(train_in[x_i]))
            self.w += predicted * train_in[x_i]
            error = self.calculate_error(train_in, y)
            if error < error_min:
                error_min = error
                w_min = self.w
            # self.w = w_min
            i += 1
            n += 1
        print(error_min)

        if i >= self.n_iterations:
            print(f"Finished due to reaching {self.n_iterations} iterations")
        return self

    def activation(self, weighted_sum):
        if self.linear:
            return weighted_sum
        return np.tanh(weighted_sum)
    
    def predict(self, input):
        return self.activation(np.dot(input, self.w)) 

    def prediction(self, input):
        ones = np.ones((input.shape[0], 1))
        input = np.concatenate((ones, input), axis=1)
        return self.activation(np.dot(input, self.w))

    def calculate_error(self, train_in, y):
        error = 0
        for xi, yi in zip(train_in, y):
            error += (yi-self.predict(xi))**2
        return error/2