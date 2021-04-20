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
count_threshold: int = config["count_threshold"]
error_threshold: float = config["error_threshold"]
epoch_training: bool = config["epoch_training"]

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
adaptive_params: tuple = tuple([dec_k, inc_k, a, b])

# read the files and get the training data, and expected out data
full_training_set, full_expected_out_set, number_class = parser.read_files(config["training_file"],
                                                                           config["expected_out_file"],
                                                                           config["system_threshold"])

# normalize expected out data if required
if config["system"] == "tanh" or config["system"] == "exp":
    full_expected_out_set = parser.normalize_data(full_expected_out_set)

# keep only a portion of the full data set for training
training_set, expected_out_set, test_training_set, test_expected_out_set \
    = parser.extract_subset(full_training_set, full_expected_out_set, config["training_ratio"])

# activation function and its derived, if derived is not used then returns always 1
act_funcs = functions.get_activation_functions(config["system"], config["beta"],
                                               config["retro_error_enhance"], number_class)

# initialize the perceptron completely
perceptron = perceptron.ComplexPerceptron(*act_funcs, config["layout"],
                                          len(training_set[0]), len(expected_out_set[0]),
                                          config["momentum"], config["momentum_alpha"])

# randomize the perceptron initial weights if needed
if randomize_w:
    perceptron.randomize_w(randomize_w_ref)

# start the training iterations

# counters and length
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

# finish only when error is 0 or reached max iterations/epochs
while error > error_threshold and i < count_threshold:

    # in case there is random w, randomize it again
    if reset_w and (n > p * reset_w_iterations):
        perceptron.randomize_w(randomize_w_ref)
        n = 0

    # for epoch training is the full training dataset, for iterative its only one random
    train_indexes = [random.randint(0, p - 1)] if not epoch_training else range(p)
    for index in train_indexes:
        perceptron.train(training_set[index], expected_out_set[index], eta, epoch_training)

    # for epoch training update the w once the epoch is finished
    if epoch_training:
        perceptron.update_w()

    # calculate error on the output laye
    new_error = perceptron.error(training_set, expected_out_set, config["retro_error_enhance"])

    # adaptive eta
    if general_adaptive_eta:
        delta_error, delta_error_dec, k, eta = functions.adaptive_eta(error-new_error, delta_error, delta_error_dec, k,
                                                                      eta, *adaptive_params)
    error = new_error

    # update or not min error
    if error < error_min:
        error_min = error

    i += 1
    n += 1

# finished, perceptron trained
# print(perceptron)
end_motive: str = "threshold error" if error <= error_threshold else "max iterations"
print(f"Training finished due to {end_motive} reached, error: {error}, min error: {error_min}, iterations: {i}")
input("\nPress enter to check the error per the given training set")
r_pos: int = 3
for data, out in zip(training_set, expected_out_set):
    print(f"in: {np.round(data, r_pos)}, "
          f"exp: {np.round(out, r_pos)}, "
          f"out: {np.round(perceptron.activation(np.array(data)), r_pos)}, "
          f"err: {np.round(perceptron.error(data, out), r_pos)}")

input("\nPress enter to check the result with the test training set")
for data, out in zip(test_training_set, test_expected_out_set):
    print(f"in: {np.round(data, r_pos)}, "
          f"exp: {np.round(out, r_pos)}, "
          f"out: {np.round(perceptron.activation(np.array(data)), r_pos)}, "
          f"err: {np.round(perceptron.error(data, out), r_pos)}")


# finished
