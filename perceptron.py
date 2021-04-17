import random
import numpy as np


# TODO: check for input and dimension consistency
class Perceptron(object):

    def __init__(self, activation_func, dimension: int):
        self.act_func = activation_func
        self.dim: int = dimension
        self.w: np.ndarray = np.zeros(dimension)

    def train(self, training_data: np.ndarray, expected_out_data: np.ndarray,
              max_iter: int, eta: float,
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
        while error > 0 and i < max_iter:

            # in case there is random w, randomize it again
            if n > p * random_iter:
                self.w = self.init_w(random_w, random_ref)
                n = 0

            # random index to analyze a training set
            x_i = random.randint(0, p - 1)

            # recalculate w by adding the delta w
            self.w += eta * self.diff_predict_expected(y[x_i], x[x_i]) * x[x_i]

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

        if i >= max_iter:
            print(f"Finished due to reaching {max_iter} iterations")

        return i, self.w

    # given the input, and the expected value, get the difference while applying the activation function
    def diff_predict_expected(self, expected, input_data: np.ndarray) -> float:
        return expected - self.predict(input_data)

    # given an already trained (or not) perceptron and a input, get the prediction value/s made
    def predict(self, input_data: np.ndarray):
        return self.act_func(np.dot(input_data, self.w))

    # calculate the error of the perceptron given a training and expected output set
    def error(self, x, y):
        aux = 0.0
        for xi, yi in zip(x, y):
            # TODO: is abs always or when?
            aux += abs(self.diff_predict_expected(yi, xi) ** 2)
        return aux / 2

    # initialize w array randomly or not
    def init_w(self, random_w: bool, ref: int) -> np.ndarray:
        if random_w:
            return np.random.uniform(-ref, ref, size=(1, self.dim))[0]
        else:
            return self.w

# def plot(self, input):
#     w_x = np.linspace(np.min(input)-1, np.max(input)+1, len(self.w))     # de donde a donde se dibuja la recta de W
#     w_y = (-np.sign(self.w[0])*((self.w[2] + self.w[0] * w_x) / self.w[1]))
#     plt.plot(w_x, w_y, color="green")
#     for e in input:
#         if e[1] > (-np.sign(self.w[0])*((self.w[2] + self.w[0] * e[0]) / self.w[1])):
#             plt.scatter(e[0],e[1],color="red")
#         else:
#             plt.scatter(e[0],e[1],color="blue")
#     plt.show()
