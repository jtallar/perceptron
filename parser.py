import math
import random

import numpy as np


def read_files(training_name: str, out_name: str, threshold: int = 1) -> (np.ndarray, np.ndarray, object):
    number_class = int
    try:
        # read, save training data
        training = np.array(parse_file_set(training_name, number_class, threshold, training=True))
    except ValueError:
        number_class = float
        training = np.array(parse_file_set(training_name, number_class, threshold, training=True))

    # read, save and if asked normalize expected output data
    output = np.array(parse_file_set(out_name, number_class, threshold, training=False))

    return training, output, number_class


def parse_file_set(file_name: str, number_class=float, threshold: int = 1, training: bool = False) -> []:
    training_file = open(file_name, "r")
    file_data = []

    # each line could have several numbers
    # initialize or not with threshold
    for line in training_file:
        line_data = [number_class(threshold)] if training else []
        # for each number, append it
        for n in line.split():
            line_data.append(number_class(n))
        file_data.append(line_data)
    return file_data


def normalize_data(data: np.ndarray) -> np.ndarray:
    return (2. * (data - np.min(data)) / np.ptp(data) - 1) * 0.9999999999


def randomize_data(full_training_data: np.ndarray, full_expected_out_data: np.ndarray) -> (np.ndarray, np.ndarray):
    aux: np.ndarray = np.c_[full_training_data.reshape(len(full_training_data), -1),
                            full_expected_out_data.reshape(len(full_expected_out_data), -1)]

    np.random.shuffle(aux)

    return aux[:, :full_training_data.size // len(full_training_data)].reshape(full_training_data.shape), \
           aux[:, full_training_data.size // len(full_training_data):].reshape(full_expected_out_data.shape)


def extract_subset(full_training_data: np.ndarray, full_expected_out_data: np.ndarray,
                   ratio: float, cross_validation_count: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    length_train: int = math.floor(len(full_training_data) * ratio)
    length_test: int = len(full_training_data) - length_train

    test_training_data = []
    test_expected_out_data = []

    for i in range(cross_validation_count * length_test, (cross_validation_count + 1) * length_test):
        # move from full list to test
        test_training_data.append(full_training_data[i])
        test_expected_out_data.append((full_expected_out_data[i]))

        # remove data from test
        training_data = np.delete(full_training_data, i, 0)
        expected_out_data = np.delete(full_expected_out_data, i, 0)

    return training_data, expected_out_data, np.array(test_training_data), np.array(test_expected_out_data)
