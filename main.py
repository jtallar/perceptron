import json
import math
import random

import numpy as np
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

# read the files and get the training data, and expected out data
training_set, expected_out_set, number_class = parser.read_files(config["training_file"], config["expected_out_file"],
                                                                 (config["system"] == "tanh") | (config["system"] == "exp"),
                                                                 config["system_threshold"])

# activation function and its derived, if derived is not used then return 1
beta: float = config["beta"]
act_funcs_dict = {
    "sign": [lambda x: np.asarray(np.sign(x), number_class),
             lambda x: 1],

    "linear": [lambda x: x,
               lambda x: 1],

    "tanh": [lambda x: np.tanh(x * beta),
             lambda x: beta * (1 - act_funcs_dict["tanh"][0](x) ** 2)],

    "exp": [lambda x: 1 / (1 + math.e ** (-2 * beta * x)),
            lambda x: 2 * beta * act_funcs_dict["exp"][0](x) * (1 - act_funcs_dict["exp"][0](x))]
}

# initialize the perceptron completely
perceptron = perceptron.ComplexPerceptron(*tuple(act_funcs_dict[config["system"]]), config["layout"],
                                          len(training_set[0]), len(expected_out_set[0]))

# randomize the perceptron initial weights if needed
perceptron.randomize_w(randomize_w_ref) if randomize_w else None

# start the training iterations

p: int = len(training_set)
i: int = 0
n: int = 0
error_min: float = np.inf
error: float = 1

# finish only when error is 0 or reached max iterations
while error > error_threshold and i < iteration_threshold:

    # in case there is random w, randomize it again
    if reset_w & n > p * reset_w_iterations:
        perceptron.randomize_w(randomize_w_ref)
        n = 0

    # random index to analyze a training set
    index = random.randint(0, p - 1)

    # train the perceptron with only the given input
    perceptron.train(training_set[index], expected_out_set[index], eta)

    # calculate error on the output laye
    error = perceptron.error(training_set, expected_out_set)

    # update or not min error
    if error < error_min:
        error_min = error

    i += 1
    n += 1

# finished, perceptron trained
print(perceptron)
print(f"error is: {error_min}, iterations: {i}")
for data, out in zip(training_set, expected_out_set):
    print(f"in: {data}, out: {perceptron.activation(np.array(data))}, err: {perceptron.error(data, out)}")
# finished
