import numpy as np
import random
import matplotlib.pyplot as plt

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
            error = self.calculate_error(train_in, y)
            if error < error_min:
                error_min = error
                w_min = self.w
            self.w = w_min
            i += 1
        if i >= self.n_iterations and self.n_iterations >= 15:
            print("Error: El conjunto de entrenamiento es inseparable")
            exit(1)
        print(i)
        return self

    def activation(self, weighted_sum):
        return np.where(weighted_sum >= 0.0, 1, -1)
        # return np.sign(weighted_sum)
    
    def predict(self, input):
        return self.activation(np.dot(input, self.w)) 

    def prediction(self, input):
        self.plot(input)
        ones = np.ones((input.shape[0], 1))
        input = np.concatenate((ones, input), axis=1)
        return self.activation(np.dot(input, self.w))

    def calculate_error(self, train_in, y):
        error = 0
        for xi, yi in zip(train_in, y):
            error += (yi-self.predict(xi))**2
        return error/2
    
    def plot(self, input):
        w_x = np.linspace(np.min(input)-1, np.max(input)+1, len(self.w))     # de donde a donde se dibuja la recta de W
        w_y = (-np.sign(self.w[0])*((self.w[2] + self.w[0] * w_x) / self.w[1]))
        plt.plot(w_x, w_y, color="green")
        for e in input:
            if e[1] > (-np.sign(self.w[0])*((self.w[2] + self.w[0] * e[0]) / self.w[1])):
                plt.scatter(e[0],e[1],color="red")
            else:
                plt.scatter(e[0],e[1],color="blue")
        plt.show()

class LinearOrNotPerceptron(object):

    def __init__(self, n_iterations=10, eta=0.01, linear=False, beta=0.5):
        self.n_iterations = n_iterations
        self.eta = eta
        self.linear = linear
        self.beta = beta
        self.ref = 1 if linear else 100 

    def train(self, train_in, y):
        if not self.linear:
            y = 2.*(y - np.min(y))/np.ptp(y)-1
        # else:
            # y = 2.*(y - np.min(y))/np.ptp(y)-1
        print(y)
        i = 0
        n=0
        error_min = np.inf
        error = 1
        self.w = np.zeros(len(train_in[0]))
        self.w = np.random.uniform(-self.ref,self.ref, size=(1,len(train_in[0])))[0]
        w_min = []
        while error > 0.001 and i < self.n_iterations:
            if n > self.n_iterations/10:
                n = 0
                self.w = np.random.uniform(-self.ref,self.ref, size=(1,len(train_in[0])))[0]
            x_i = random.randint(0,len(train_in)-1)
            predicted = self.eta * (y[x_i] - self.predict(train_in[x_i]))
            self.w += predicted * train_in[x_i]
            error = self.calculate_error(train_in, y)
            if error < error_min:
                error_min = error
                w_min = self.w
            self.w = w_min
            i += 1
            n += 1

        if i >= self.n_iterations:
            print(f"Finished due to reaching {self.n_iterations} iterations")
        return self

    def activation(self, weighted_sum):
        if self.linear:
            return weighted_sum
        return np.tanh(self.beta*weighted_sum)
    
    def predict(self, input):
        return self.activation(np.dot(input, self.w)) 

    def prediction(self, input):
        ones = np.ones((input.shape[0], 1))
        input = np.concatenate((ones, input), axis=1)
        return self.activation(np.dot(input, self.w))

    def calculate_error(self, train_in, y):
        error = 0
        for xi, yi in zip(train_in, y):
            error += abs((yi-self.predict(xi))**2)
        return error/2