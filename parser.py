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
