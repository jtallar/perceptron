import json
import math
import random

import numpy as np

import functions
import parser
import perceptron
import metrics

with open("config.json") as file:
    config = json.load(file)

# static non changeable vars
cross_validation: bool = config["cross_validation"]
training_ratio: float = config["training_ratio"]
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

# activation function and its derived, if derived is not used then returns always 1
act_funcs = functions.get_activation_functions(config["system"], config["beta"],
                                               config["retro_error_enhance"], number_class)

# randomize input distribution, used for cross validation
if cross_validation:
    full_training_set, full_expected_out_set = parser.randomize_data(full_training_set, full_expected_out_set)

# for cross validations
cross_validation_count: int = 1 if not cross_validation else math.floor(1 / (1 - training_ratio))
j: int = 0

# for metrics
best_appreciation: float = 0
delta_eq: float = config["delta_metrics"]
best_acc_train: float = np.inf
best_acc_test: float = np.inf
best_err_train: float = np.inf
best_err_test: float = np.inf
best_perceptron: perceptron.ComplexPerceptron


# do only one if it is not cross validation
while j < cross_validation_count:
    eta: float = config["eta"]

    # keep only a portion of the full data set for training
    training_set, expected_out_set, test_training_set, test_expected_out_set \
        = parser.extract_subset(full_training_set, full_expected_out_set, training_ratio, j)

    # initialize the perceptron completely
    c_perceptron = perceptron.ComplexPerceptron(*act_funcs, config["layout"],
                                                len(training_set[0]), len(expected_out_set[0]),
                                                config["momentum"], config["momentum_alpha"])

    # randomize the perceptron initial weights if needed
    if randomize_w:
        c_perceptron.randomize_w(randomize_w_ref)

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
            c_perceptron.randomize_w(randomize_w_ref)
            n = 0

        # for epoch training is the full training dataset, for iterative its only one random
        train_indexes = [random.randint(0, p - 1)] if not epoch_training else range(p)
        for index in train_indexes:
            c_perceptron.train(training_set[index], expected_out_set[index], eta, epoch_training)

        # for epoch training update the w once the epoch is finished
        if epoch_training:
            c_perceptron.update_w()

        # calculate error on the output laye
        new_error = c_perceptron.error(training_set, expected_out_set, config["retro_error_enhance"])

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

    # get metrics and error values, save the best perceptron
    acc_train = metrics.accuracy(c_perceptron.activation(training_set), expected_out_set, delta_eq)
    err_train = c_perceptron.error(training_set, expected_out_set)
    acc_test = metrics.accuracy(c_perceptron.activation(test_training_set), test_expected_out_set, delta_eq)
    err_test = c_perceptron.error(test_training_set, test_expected_out_set)
    appreciation = metrics.appreciation(acc_train, err_train, acc_test, err_test)
    if appreciation >= best_appreciation:
        best_acc_train, best_acc_test, best_err_train, best_err_test = (acc_train, acc_test, err_train, err_test)
        best_appreciation = appreciation
        best_perceptron = c_perceptron

    j += 1


print(f"Training set accuracy: {best_acc_train}, test set accuracy: {best_acc_test}")
print(f"Training set error: {best_err_train}, test set error: {best_err_test}")
print(best_perceptron)

# finished
