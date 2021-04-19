import json
import random

import numpy as np

import functions
import parser

# Read configurations from file
import perceptron

with open("config.json") as file:
    config = json.load(file)

# TODO: check for errors
eta: float = config["eta"]
iteration_threshold: int = config["iteration_threshold"]
error_threshold: float = config["error_threshold"]

# w randomization and reset configs
randomize_w: bool = config["randomize_w"]
randomize_w_ref: int = config["randomize_w_ref"]
reset_w: bool = config["reset_w"]
reset_w_iterations: int = config["reset_w_iterations"]

# adaptive eta configurations
general_adaptive_eta: bool = config["general_adaptive_eta"]
a: float = config["a"]
dec_k: int = config["delta_error_decrease_iterations"]
b: float = config["b"]
inc_k: int = config["delta_error_increase_iterations"]

# read the files and get the training data, and expected out data
training_set, expected_out_set, number_class = parser.read_files(config["training_file"], config["expected_out_file"],
                                                                 config["system_threshold"])

# normalize expected out data if required
expected_out_set = parser.normalize_data(expected_out_set) if (config["system"] == "tanh") | \
                                                              (config["system"] == "exp") else None

# activation function and its derived, if derived is not used then returns always 1
act_funcs = functions.get_activation_functions(config["system"], config["beta"],
                                               config["retro_error_enhance"], number_class)

# initialize the perceptron completely
perceptron = perceptron.ComplexPerceptron(*act_funcs, config["layout"],
                                          len(training_set[0]), len(expected_out_set[0]),
                                          config["momentum"], config["momentum_alpha"])

# randomize the perceptron initial weights if needed
perceptron.randomize_w(randomize_w_ref) if randomize_w else None

# start the training iterations

# counters and lenght
p: int = len(training_set)
i: int = 0
n: int = 0

# error handling
error: float = np.inf
error_min: float = np.inf

# adaptive eta vars
delta_error: float = np.inf
delta_error_dec: bool = True
k: int = 0

# finish only when error is 0 or reached max iterations
while error > error_threshold and i < iteration_threshold:

    # in case there is random w, randomize it again
    if reset_w & n > p * reset_w_iterations:
        perceptron.randomize_w(randomize_w_ref)
        n = 0

    # random index to analyze a training set
    index = random.randint(0, p - 1)

    # TODO: adaptive eta per neuron
    # train the perceptron with only the given input
    perceptron.train(training_set[index], expected_out_set[index], eta)

    # calculate error on the output laye
    new_error = perceptron.error(training_set, expected_out_set, config["retro_error_enhance"])

    # update or not min error
    if error < error_min:
        error_min = error

    # adaptive eta
    if general_adaptive_eta:
        new_delta_error: float = error - new_error
        # the delta error changes direction, reset k (-1 +1 = 0)
        if (delta_error > new_delta_error) != delta_error_dec:
            k = -1
        if (k == dec_k) & delta_error_dec:
            eta += a
        if (k == inc_k) & (not delta_error_dec):
            eta -= b * eta

        delta_error_dec = delta_error > new_delta_error
        delta_error = new_delta_error
        k += 1

    error = new_error
    i += 1
    n += 1

# finished, perceptron trained
print(perceptron)
print(f"error is: {error_min}, iterations: {i}")
for data, out in zip(training_set, expected_out_set):
    print(f"in: {data}, expected: {out}, out: {perceptron.activation(np.array(data))}, err: {perceptron.error(data, out)}")
# finished
