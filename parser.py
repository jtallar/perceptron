import numpy as np


def read_files(training_name: str, out_name: str, number_class=float,
               normalize_o: bool = False) -> (np.ndarray, np.ndarray):

    # read, save training data
    training = np.array(read_training_set(training_name, number_class))

    # read, save and if asked normalize expected output data
    output = np.array(read_desired_output(out_name, number_class))
    if normalize_o:
        output = normalize_data(output)

    return training, output


def read_training_set(file_name: str, number_class=float) -> []:
    training_file = open(file_name, "r")
    training = []
    # each line has several numbers
    # initialize with 1 for later multiplication with w0
    for line in training_file:
        data = [number_class(1)]
        # for each number, append it
        for n in line.split():
            data.append(number_class(n))
        training.append(data)
    return training


def read_desired_output(file_name: str, number_class=float) -> []:
    output_file = open(file_name, "r")
    output = []
    for line in output_file:
        # append the number of the first item in line split
        # there is only one
        output.append(number_class(line.split()[0]))
    return output


def normalize_data(data: np.ndarray) -> np.ndarray:
    return 2. * (data - np.min(data)) / np.ptp(data) - 1
