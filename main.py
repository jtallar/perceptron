import json
import math

import numpy as np

import parser
import perceptron


# Read configurations from file
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
training, outputs = parser.read_files(training_file_name, expected_out_file_name,
                                      number_class, normalize_expected_out_file, threshold)

# initialize perceptron
perceptron = perceptron.SimplePerceptron(activation_functions, len(training[0]))

# train perceptron and print step count and w
print(perceptron.train(training, outputs, max_iter, eta, error_threshold,
                       randomize_w, randomize_w_refs, change_w_iterations))

# print difference between the real output and the one from the perceptron
print(perceptron.diff_predict_expected(outputs, training))

# plot the graphic with the training set
perceptron.plot(training, steps_graph_3d)

