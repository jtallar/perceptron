import math

def get_metrics(perceptron, training_set, expected_out, test_set, expected_test_out, delta): # data, expected_out, activation(data), delta(para el rango)
    
    train_error, train_success = 0, 0
    for data, out in zip(training_set, expected_out):
        activated = perceptron.activation(data)
        if math.isclose(activated, out, rel_tol=delta):
            train_success+=1
        train_error+=perceptron.error(data, out)
    train_accuracy = train_success / len(training_set)

    test_error, test_success = 0, 0
    for data, out in zip(test_set, expected_test_out):
        activated = perceptron.activation(data)
        if math.isclose(activated, out, rel_tol=delta):
            test_success+=1
        test_error+=perceptron.error(data, out)
    test_accuracy = test_success / len(test_set)

    valoration = 0.5*train_accuracy + 0.5*test_accuracy

    return train_accuracy, train_error, test_accuracy, test_error, valoration