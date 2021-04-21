import math

def get_metrics(in_set, expected_out, delta): # data, expected_out, activation(data), delta(para el rango)
    success = 0
    for data, out in zip(in_set, expected_out):
        if math.isclose(data, out, rel_tol=delta):
            success+=1
    return success / len(in_set)

def get_appreciation(train_accuracy, train_error, test_accuracy, test_error):
    return 0.5*train_accuracy + 0.5*test_accuracy
