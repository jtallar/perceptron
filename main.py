import json
import math
import random

import numpy as np
import parser

# Read configurations from file
import perceptron
import perceptron1

with open("config.json") as file:
    config = json.load(file)

# TODO: check for errors
max_iter: int = config["max_iter"]
eta: float = config["eta"]
beta: float = config["beta"]

training_file_name: str = config["training_file"]
expected_out_file_name: str = config["expected_out_file"]
number_class = float if config["float_data"] else int
threshold: int = config["threshold"]
error_threshold: float = config["error_threshold"]

randomize_w: bool = config["randomize_w"]
randomize_w_refs: int = config["randomize_w_refs"]
change_w_iterations: int = config["change_w_iterations"]

complex_layout: [int] = config["complex_layout"]

steps_graph_3d: int = config["steps_graph_3d"]

activation_functions_dict = {
    "sign": [lambda x: np.asarray(np.sign(x), number_class),
             lambda x: 1],

    "linear": [lambda x: x,
               lambda x: 1],

    "tanh": [lambda x: np.tanh(x * beta),
             lambda x: beta * (1 - activation_functions_dict["tanh"][0](x) ** 2)],

    "exp": [lambda x: 1 / (1 + math.e ** (-2 * beta * x)),
            lambda x: 2 * beta * activation_functions_dict["exp"][0](x) * (1 - activation_functions_dict["exp"][0](x))]

}

# activation functions and boolean for normalize
activation_functions = activation_functions_dict[config["system"]]
normalize_expected_out_file: bool = (config["system"] == "tanh") | (config["system"] == "exp")

# read the files
training_data, expected_out_data = parser.read_files(training_file_name, expected_out_file_name,
                                                     number_class, normalize_expected_out_file, threshold)

# initialize the perceptron completely
perceptron = perceptron.ComplexPerceptron(*tuple(activation_functions), complex_layout, len(training_data[0]))

p: int = len(training_data)
i: int = 0
error_min: float = np.inf
error: float = 1

# finish only when error is 0 or reached max iterations
while error > error_threshold and i < max_iter:

    # random index to analyze a training set
    index = random.randint(0, p - 1)

    # train the perceptron with only the given input
    perceptron.train(training_data[index], expected_out_data[index], eta)

    # calculate error on the output laye
    error = perceptron.error(training_data, expected_out_data)

    # update or not min error
    if error < error_min:
        error_min = error

    i += 1

print(perceptron)
print(f"error is: {error_min}, iterations: {i}")
for data in training_data:
    print(perceptron.activation(np.array(data)))
# finished


# # initialize perceptron ONLY SIMPLE
# perceptron = perceptron1.SimplePerceptron(activation_functions, len(training_data[0]))
#
# # train perceptron and print step count and w
# print(perceptron.train(training_data, expected_out_data, max_iter, eta, error_threshold,
#                        randomize_w, randomize_w_refs, change_w_iterations))
#
# # print difference between the real output and the one from the perceptron
# print(perceptron.diff_predict_expected(expected_out_data, training_data))
#
# # plot the graphic with the training set
# perceptron.plot(training_data, steps_graph_3d)
