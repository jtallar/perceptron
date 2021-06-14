import multiprocessing.pool
import numpy as np


class SimplePerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 dimension: int, hidden: bool = False, index: int = 0,
                 momentum: bool = False, mom_alpha: float = 0.9):
        self.index = index
        self.hidden: bool = hidden
        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.w: np.ndarray = np.zeros(dimension)
        self.input: np.ndarray = np.zeros(dimension)

        # momentum correction data
        self.prev_delta_w = np.zeros(dimension)
        self.momentum: bool = momentum
        self.mom_alpha: float = mom_alpha

        # for non iterative training (epoch)
        self.accu_w = np.zeros(dimension)

    # out, a 1D array, is used only in the most superior layer
    # sup_w is a 2D matrix with all the W vectors of the superior layer
    # sup_delta is a 1D array, resulting in all the delta values of the superior layer
    # the two above are only used in hidden layers
    def train(self, out: np.ndarray, sup_w: np.ndarray, sup_delta: np.ndarray, eta: float, epoch: bool = False) \
            -> (np.ndarray, float):
        # activation for this neuron
        activation_derived = self.act_func_der(np.dot(self.input, self.w))

        # delta sub i using the activation values
        if not self.hidden:
            delta = (out[self.index] - self.activation(self.input)) * activation_derived
        else:
            delta = np.dot(sup_delta, sup_w[:, self.index]) * activation_derived

        # calculate the delta w
        delta_w = (eta * delta * self.input)

        if not epoch:
            # for iterative update
            self.update_w(delta_w=delta_w, epoch=False)
        else:
            # epoch training accumulation
            self.accu_w += delta_w

        return self.w, delta

    # returns the activation value/s for the given input in this neuron
    # returns int or float depending on the input data and activation function
    def activation(self, input_arr: np.ndarray, training: bool = False):
        if training:
            self.input = input_arr

        # activation for this neuron, could be int or float, or an array in case is the full dataset
        return self.act_func(np.dot(input_arr, self.w))

    # calculates the error given the full training dataset
    def error(self, inp: np.ndarray, out: np.ndarray) -> float:
        return np.sum(np.abs((out - self.activation(inp)) ** 2)) / 2

    # resets the w to a randomize range
    def randomize_w(self, ref: float) -> None:
        self.w = np.random.uniform(-ref, ref, len(self.w))

    # for epoch training delta is the accum value
    # for iterative training is the delta of each time
    def update_w(self, delta_w: np.ndarray = np.asarray([]), epoch: bool = False):
        if epoch:
            delta_w = self.accu_w

        self.w += delta_w

        # in case of momentum, calculate delta w and update values
        if self.momentum:
            self.w += self.mom_alpha * self.prev_delta_w
            self.prev_delta_w = delta_w

    def __str__(self) -> str:
        return f"SP=(i={self.index}, w={self.w})"

    def __repr__(self) -> str:
        return f"SP=(i={self.index}, w={self.w})"


class ComplexPerceptron(object):

    def __init__(self, activation_function, activation_function_derived,
                 layout: [int], input_dim: int, output_dim: int,
                 momentum: bool = False, mom_alpha: float = 0.9):

        self.act_func = activation_function
        self.act_func_der = activation_function_derived
        self.network = None
        self.__init_network(layout, input_dim, output_dim, momentum, mom_alpha)

    # train with the input dataset the complex perceptron
    def train(self, training_set: np.ndarray, expected_out: np.ndarray, eta: float = 0.01, epoch: bool = False) \
            -> None:

        # propagate activation values while saving the input data, first one is training set
        self.activation(training_set, training=True)

        # retro propagate the delta
        sup_w: np.ndarray = np.empty(1)
        sup_delta: np.ndarray = np.empty(1)
        for layer in reversed(self.network):
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            sup_w, sup_delta = zip(*pool.map(lambda s_p: s_p.train(expected_out, sup_w, sup_delta, eta, epoch), layer))
            # convert tuples to lists
            sup_w = np.asarray(sup_w)
            sup_delta = np.asarray(sup_delta)

    # propagates input along the entire network
    # in case of training, saves  the input for later computation on retro propagation
    # returns the final activation value
    def activation(self, init_input: np.ndarray, training: bool = False) -> np.ndarray:
        activation_values = init_input
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            activation_values = pool.map(lambda s_p: s_p.activation(activation_values, training=training), layer)
            activation_values = np.transpose(np.asarray(activation_values))

        return activation_values

    # calculate the error on the perceptron
    def error(self, inp: np.ndarray, out: np.ndarray, error_enhance: bool = False) -> float:
        if not error_enhance:
            return (np.linalg.norm(out[:, 1:] - self.activation(inp)[:, 1:]) ** 2) / len(out[:, 1:])


        return np.sum((1 + out) * np.log(np.divide((1 + out), (1 + self.activation(inp)))) / 2 +
                      (1 - out) * np.log(np.divide((1 - out), (1 - self.activation(inp)))) / 2)

    # resets the w to a randomize range if desired for the entire network
    # if randomize is false, then does nothing
    def randomize_w(self, ref: float) -> None:
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            pool.map(lambda s_p: s_p.randomize_w(ref), layer)

    # for epoch training updates each w with its accum
    def update_w(self) -> None:
        for layer in self.network:
            pool = multiprocessing.pool.ThreadPool(processes=len(layer))
            pool.map(lambda s_p: s_p.update_w(epoch=True), layer)

    def __str__(self) -> str:
        out: str = "CPerceptron=("
        for i, layer in enumerate(self.network):
            out += f"\nlayer {i}=" + str(layer)
        return out + ")"

    def __repr__(self) -> str:
        out: str = "CPerceptron=("
        for i, layer in enumerate(self.network):
            out += f"\nlayer {i}=" + str(layer)
        return out + ")"

    # private methods

    # initializes the entire network of perceptron given a layout
    def __init_network(self, hidden_layout: [int], input_dim: int, output_dim: int,
                       momentum: bool = False, mom_alpha: float = 0.9) -> None:
        # the final amount of perceptron depends on expected output dimension
        layout: np.ndarray = np.append(np.array(hidden_layout, dtype=int), output_dim)

        # initialize the length of the network
        self.network = np.empty(shape=len(layout), dtype=np.ndarray)

        # for each level, get its count of perceptron
        for level in range(len(layout)):

            # initialize (empty) level with its amount of perceptron
            self.network[level] = np.empty(shape=layout[level], dtype=SimplePerceptron)

            # the dimension of the next level is set from the previous or the input data
            dim: int = layout[level - 1] if level != 0 else input_dim

            # create the corresponding amount of perceptron
            for index in range(layout[level]):
                # for each index and level, create the corresponding perceptron
                self.network[level][index] = \
                    SimplePerceptron(self.act_func, self.act_func_der, dim,
                                     level != len(layout) - 1, index, momentum, mom_alpha)
