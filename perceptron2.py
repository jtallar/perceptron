import multiprocessing.pool
import queue
from multiprocessing
import random

import numpy as np

import perceptron


class SimplePerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 hidden: bool, dimension: int, index: int):
        self.index = index
        self.hidden: bool = hidden
        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.w: np.ndarray = np.zeros(dimension)

    # input is the input 1D array, as is a copy of the single out values from the inferior layer
    # sup_aux is a 1D array, resulting in all the phi? values of the superior layer
    # sup_w is a 2D matrix with all the W vectors of the superior layer
    # the two above are only used in hidden layers
    # out is used only in the most superior layer
    def train(self, eta: float, inp: np.ndarray, sup_aux: np.ndarray, sup_w: np.ndarray, out):
        # activation for this neuron
        activation_derived = self.activation_derived(inp)

        # phi? sub i using the activation values
        if self.hidden:
            aux = (out - self.activation(inp)) * activation_derived
        else:
            aux = np.dot(sup_aux, sup_w[:, self.index]) * activation_derived

        self.w += (eta * aux * inp)
        return self.w, aux

    # returns the activation value/s for the given input in this neuron
    # returns int or float depending on the input data and activation function
    def activation(self, input_arr: np.ndarray):
        # activation for this neuron, could be int or float, or an array in case is the full dataset
        return self.act_func(np.dot(input_arr, self.w))

    # returns the derived activation value/s for the given input in this neuron
    # returns int or float depending on the input data and activation function
    def activation_derived(self, input_arr: np.ndarray):
        # activation for this neuron
        return self.act_func_der(np.dot(input_arr, self.w))

    # calculates the error given the full training dataset
    def error(self, inp: np.ndarray, out: np.ndarray):
        return np.sum(np.abs((out - self.activation(inp)) ** 2)) / 2

    def index(self) -> int:
        return self.index



class ComplexPerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], dimension: int, eta: float):
        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.dim = dimension


        # TODO: create simple perceptron given the layout
        for layer_level in np.arange(0, len(layout)):
            for layer_index in np.arange(0, layout[layer_level]):
                self.layout[layer_level][layer_index] = SimplePerceptron(activation_function, activation_function_derived,
                                                                         True, dimension, layer_index)

    def train(self, training_data: np.ndarray, expected_out_data: np.ndarray,
              max_iter: int, error_threshold: float = 0.0):
        # rename and initialize some data
        x: np.ndarray = training_data
        y: np.ndarray = expected_out_data

        p: int = len(x)
        i: int = 0
        error_min: float = np.inf
        error: float = 1

        # finish only when error is 0 or reached max iterations
        while error > error_threshold and i < max_iter:
            # random index to analyze a training set
            x_i = random.randint(0, p - 1)

            # the activation values, they start off as the chosen training set
            activation_values: x[x_i]
            for layer in self.layout:
                pool = multiprocessing.pool.ThreadPool(processes=len(layer))
                activation = pool.map(lambda s_p: (s_p.index(), s_p.activation(activation_values)), layer)
                print(activation)

            exit(1)
            # calculate error on the output layer (take the max of all the posibles)
            error = 2

            # update or not min error
            if error < error_min:
                error_min = error

            # the previous val if new error is not min, otherwise the new one
            i += 1

        return i, error










