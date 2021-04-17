import sys
import json
import numpy as np
import perceptronJuli
import parser

# Read configurations from file
with open("config.json") as file:
    config = json.load(file)

# TODO: proteger de error / inexistente
max_iter = config["max_iter"]
eta = config["eta"]
training_linear = config["training_file"]
out_linear = config["expected_out"]
is_linear = True if config["linear"]=="true" else False
beta = config["beta"]

# Testing Simple

inputs = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
outputs = np.array([1, -1, -1, -1])
perceptron = perceptronJuli.SimplePerceptron(max_iter, eta)
perceptron.train(inputs, outputs)
print(perceptron.prediction(inputs)) 


# Testing Linear or Non-Linear
'''
perceptron = perceptron.LinearOrNotPerceptron(max_iter, eta, is_linear, beta)
t, o = parser.training(training_linear, out_linear)
perceptron.train(t, o)
inputs = np.array([[4.4793, -4.0765, 4.4558], [-4.1793, -4.9218, 1.7664], [-3.9429, -0.7689, 4.8830], [-3.5796,1.5557,2.6683]])
print(perceptron.prediction(inputs)) 
'''
# Testing Other
'''
inputs = [[1.0,1.0,0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0 , 1.0]]
perceptron = linear.LinearOrNotPerceptron(max_iter, eta)
perceptron.train(inputs, outputs)

print(perceptron.prediction(np.array(inputs))) 
'''