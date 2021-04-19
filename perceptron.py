import multiprocessing.pool
import numpy as np


class SimplePerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 dimension: int, hidden: bool = False, index: int = 0):
        self.index = index
        self.hidden: bool = hidden
        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.w: np.ndarray = np.random.uniform(-1.0, 1.0, dimension)
        self.input: np.ndarray = np.zeros(dimension)

    # out is used only in the most superior layer
    # sup_w is a 2D matrix with all the W vectors of the superior layer
    # sup_aux is a 1D array, resulting in all the phi? values of the superior layer
    # the two above are only used in hidden layers
    def train(self, out, sup_w: np.ndarray, sup_aux: np.ndarray, eta: float):
        # activation for this neuron
        activation_derived = self.activation_derived(self.input)

        # phi? sub i using the activation values
        if not self.hidden:
            aux = (out - self.activation(self.input)) * activation_derived
        else:
            aux = np.dot(sup_aux, sup_w[:, self.index]) * activation_derived

        self.w += (eta * aux * self.input)

        return self.w, aux

    # returns the activation value/s for the given input in this neuron
    # returns int or float depending on the input data and activation function
    def activation(self, input_arr: np.ndarray, training: bool = False):
        if training:
            self.input = np.asarray(input_arr)

        # activation for this neuron, could be int or float, or an array in case is the full dataset
        return self.act_func(np.dot(self.input, self.w))

    # returns the derived activation value/s for the given input in this neuron
    # returns int or float depending on the input data and activation function
    def activation_derived(self, input_arr: np.ndarray):
        # activation for this neuron
        return self.act_func_der(np.dot(input_arr, self.w))

    # calculates the error given the full training dataset
    def error(self, inp: np.ndarray, out: np.ndarray):
        return np.sum(np.abs((out - self.activation(inp)) ** 2)) / 2

    def get_index(self) -> int:
        return self.index

    def __str__(self):
        return f"SPerceptron=(index={self.index}, hidden={self.hidden}, w={self.w})"

    def __repr__(self):
        return f"SPerceptron=(index={self.index}, hidden={self.hidden}, w={self.w})"


class ComplexPerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], input_dim: int):

        self.act_func = activation_function
        self.act_func_der = activation_function_derived

        # initialize (empty) the general array with layout length
        self.network = np.empty(shape=len(layout), dtype=np.ndarray)
        self.init_network(layout, input_dim)

    # train with the input dataset the complex perceptron
    def train(self, training_set: np.ndarray, expected_out, eta: float = 0.01):

        # propagate activation values while saving the input data, first one is training set
        self.activation(training_set, training=True)

        # retro propagate the phi?
        sup_w: np.ndarray = np.empty(1)
        sup_aux: np.ndarray = np.empty(1)
        for layer in reversed(self.network):
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            sup_w, sup_aux = zip(*pool.map(lambda s_p: s_p.train(expected_out, sup_w, sup_aux, eta), layer))
            # convert tuples to lists
            sup_w = np.asarray(sup_w)
            sup_aux = np.asarray(sup_aux)

    # propagates input along the entire network
    # in case of training, saves the input for later computation on retro propagation
    # returns the final activation value
    def activation(self, init_input: np.ndarray, training: bool = False):
        activation_values = init_input
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            activation_values = pool.map(lambda s_p: s_p.activation(activation_values, training=training), layer)

        return activation_values[0]

    # calculate the error on the perceptron
    def error(self, inp: np.ndarray, out: np.ndarray) -> float:
        return np.sum(np.abs((out - self.activation(inp)) ** 2)) / 2

    # initializes the entire network of perceptron given a layout
    # it is assumed that the network is defined in size by the amount of levels
    def init_network(self, layout: [int], input_dim: int) -> None:
        # for each level, get its count of perceptron
        for level in range(len(layout)):

            # initialize (empty) level with its amount of perceptron
            self.network[level] = np.empty(shape=layout[level], dtype=SimplePerceptron)
            dim: int = layout[level - 1] if level != 0 else input_dim

            # create the corresponding amount of perceptron
            for index in range(layout[level]):
                # for each index and level, create the corresponding perceptron
                self.network[level][index] = \
                    SimplePerceptron(self.act_func, self.act_func_der,
                                     dim, level != len(layout) - 1, index)

    def __str__(self):
        return f"CPerceptron=({self.network})"

    def __repr__(self):
        return f"CPerceptron=({self.network})"
