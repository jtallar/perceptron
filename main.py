import json
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

normalize_expected_out_file: bool = config["normalize_expected_out_file"]

randomize_w: bool = config["randomize_w"]
randomize_w_refs: int = config["randomize_w_refs"]
change_w_iterations: int = config["change_w_iterations"]

activation_functions_dict = {
    "sign": lambda x: np.asarray(np.sign(x), number_class),
    "linear": lambda x: x,
    "nonlinear": lambda x: np.tanh(x * beta)
}
activation_function = activation_functions_dict[config["system"]]


# read the files
training, outputs = parser.read_files(training_file_name, expected_out_file_name,
                                      number_class, normalize_expected_out_file, threshold)

# initialize perceptron
perceptron = perceptron.Perceptron(activation_function, len(training[0]))

# train perceptron and print step count and w
print(perceptron.train(training, outputs, max_iter, eta,
                       randomize_w, randomize_w_refs, change_w_iterations))

# print difference between the real output and the one from the perceptron
print(perceptron.diff_predict_expected(outputs, training))

