import math
import random

import matplotlib.pyplot as plt
import numpy as np


# TODO: check for input and dimension consistency
class SimplePerceptron(object):

    def __init__(self, activation_functions: [], dimension: int):
        self.act_func = activation_functions[0]
        self.act_func_der = activation_functions[1]
        self.dim: int = dimension
        self.w: np.ndarray = np.zeros(dimension)

    def train(self, training_data: np.ndarray, expected_out_data: np.ndarray,
              max_iter: int, eta: float, error_threshold: float = 0.0,
              random_w: bool = False, random_ref: int = 100, random_iter: int = 100):

        # rename and initialize some data
        x: np.ndarray = training_data
        y: np.ndarray = expected_out_data
        self.w = self.init_w(random_w, random_ref)

        p: int = len(x)
        i: int = 0
        n: int = 0
        error_min: float = np.inf
        error: float = 1
        w_min: np.ndarray = self.w

        # finish only when error is 0 or reached max iterations
        while error > error_threshold and i < max_iter:

            # in case there is random w, randomize it again
            if n > p * random_iter:
                self.w = self.init_w(random_w, random_ref)
                n = 0

            # random index to analyze a training set
            x_i = random.randint(0, p - 1)

            # recalculate w by adding the delta w using activation derived and not
            self.w += eta * (y[x_i] - self.predict(x[x_i])) * self.predict_der(x[x_i]) * x[x_i]

            # calculate error with the current w
            error = self.error(x, y)

            # update or not min error and min w
            if error < error_min:
                error_min = error
                w_min = self.w

            # the previous val if new error is not min, otherwise the new one
            self.w = w_min
            i += 1
            n += 1

        return i, error, self.w

    # given an already trained (or not) perceptron and a input, get the prediction value/s made
    def predict(self, input_data: np.ndarray):
        return self.act_func(np.dot(input_data, self.w))

    # predict derivative of the activation function
    def predict_der(self, input_data: np.ndarray):
        return self.act_func_der(np.dot(input_data, self.w))

    # calculate the error of the perceptron given a training and expected output set
    def error(self, x, y):
        aux = 0.0
        for xi, yi in zip(x, y):
            aux += abs((yi - self.predict(xi)) ** 2)
        return aux / 2

    # initialize w array randomly or not
    def init_w(self, random_w: bool, ref: int) -> np.ndarray:
        if random_w:
            return np.random.uniform(-ref, ref, size=(1, self.dim))[0]
        else:
            return self.w

    # given the input, and the expected value, get the difference while applying the activation function
    def diff_predict_expected(self, expected, input_data: np.ndarray) -> float:
        return expected - self.predict(input_data)

    # plot the solution
    def plot(self, input_data: np.ndarray, steps: int = 20) -> ():

        # check for valid plotting dimension
        if 2 < self.dim - 1 > 3:
            print(f"Cannot plot a {self.dim - 1} dimension perceptron")
            exit(1)

        def function(xi, yi, *argv):
            return (self.w[0] + self.w[1] * xi + self.w[2] * yi) / (-self.w[-1])

        # set the figure dimensions and axis labels
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if self.dim - 1 == 3 else None)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # get the top values for dim - 1 axis
        max_array: np.ndarray = np.abs(input_data[0])
        for point in input_data:
            # do not take w0 and last one
            for i in np.arange(1, len(point) - 1, 1):
                if abs(point[i]) > max_array[i]:
                    max_array[i] = abs(point[i])
        # delete first (w0) and last
        max_array = np.delete(max_array, [0, -1])

        # generate range for dim - 1 axis
        args_w_plot = []
        for max_val in max_array:
            # add a 10% of margin and grab ceiling
            max_val: float = math.ceil(max_val * 1.15)
            # divide by X the range
            args_w_plot.append(np.linspace(-max_val, max_val, steps))

        if self.dim - 1 == 3:
            ax.set_zlabel('Z')
            x, y = np.meshgrid(*tuple(args_w_plot))
            z = function(x, y)
            ax.plot_wireframe(x, y, z)
        else:
            x = args_w_plot[0]
            y = function(x, 0)
            ax.plot(x, y)

        # print each point in the graphic
        for point in input_data:
            p_args: tuple = tuple(np.delete(point, 0))

            if np.array(p_args)[-1] > function(*p_args, 0):
                ax.scatter(*p_args, color="red")
            else:
                ax.scatter(*p_args, color="blue")

        # plot
        plt.show()
