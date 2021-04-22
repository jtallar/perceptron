import math

import numpy as np

import perceptron


def function(x, trust) -> int:
    if x >= trust:
        return 1
    if x <= -trust:
        return -1
    return 0


def discrete(data: np.ndarray, trust: float) -> np.ndarray:
    return np.vectorize(function)(data, trust)


def accuracy(in_set, expected_out, delta):  # data, expected_out, activation(data), delta(para el rango)
    success = 0
    for data, out in zip(in_set, expected_out):
        if math.isclose(data, out, rel_tol=delta):
            success += 1
    return success / len(in_set)


def appreciation(train_accuracy, train_error, test_accuracy, test_error):
    return 0.5 * train_accuracy + 0.5 * test_accuracy


def appreciation_single(train_accuracy, train_error):
    return train_accuracy


def metrics(old_metrics: dict, p: perceptron.ComplexPerceptron, training_set: np.ndarray, expected_out_set: np.ndarray,
            test_training_set: np.ndarray, test_expected_out_set: np.ndarray, delta_eq: float, normalize: bool,
            trust: float) -> (dict, dict):

    if normalize:
        train_predicted = discrete(p.activation(training_set), trust)
    else:
        train_predicted = p.activation(training_set)
    acc_train = accuracy(train_predicted, expected_out_set, delta_eq)
    err_train = p.error(training_set, expected_out_set)
    appreciation_val = appreciation_single(acc_train, err_train)
    test_predicted = []
    acc_test = 0
    err_test = 0

    # in case is a full training set then do not attempt this
    if len(test_training_set) != 0:
        if normalize:
            test_predicted = discrete(p.activation(test_training_set), trust)
        else:
            test_predicted = p.activation(test_training_set)
        acc_test = accuracy(test_predicted, test_expected_out_set, delta_eq)
        err_test = p.error(test_training_set, test_expected_out_set)
        appreciation_val = appreciation(acc_train, err_train, acc_test, err_test)

    new_metrics = {
        "acc_train": acc_train,
        "err_train": err_train,
        "acc_test": acc_test,
        "err_test": err_test,

        "appreciation": appreciation_val,
        "perceptron": p,

        "training_set": training_set,
        "expected_out_set": expected_out_set,
        "test_training_set": test_training_set,
        "test_expected_out_set": test_expected_out_set,

        "train_predicted": train_predicted,
        "test_predicted": test_predicted,

        "normalized": normalize
    }

    if "appreciation" in old_metrics and old_metrics["appreciation"] > appreciation_val:
        return old_metrics, new_metrics

    return new_metrics, new_metrics
