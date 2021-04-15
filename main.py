import sys
import parser 
import json
import numpy as np
import simple
import linear
import parser

# Read configurations from file
with open("config.json") as file:
    config = json.load(file)

max_iter = config["max_iter"]
eta = config["eta"]
training_linear = config["training_file"]
out_linear = config["expected_out"]


inputs = np.array([[1,1], [1,0], [0,1], [0,0]])
outputs = np.array([1, 0, 0, 1])
perceptron = simple.SimplePerceptron(max_iter, eta)
perceptron.train(inputs, outputs)
print(perceptron.prediction(inputs)) 


"""
t, o = parser.training(training_linear, out_linear)
perceptron = linear.SimplePerceptron(max_iter, eta)
perceptron.train(t, o)
inputs = np.array([[4.4793, -4.0765, 4.4558], [-4.1793, -4.9218, 1.7664], [-3.9429, -0.7689, 4.8830], [-3.5796,1.5557,2.6683]])
print(perceptron.prediction(inputs)) 
"""


# inputs = [[1.0,1.0,0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0 , 1.0]]
# perceptron = linear.SimplePerceptron(max_iter, eta)
# perceptron.train(inputs, outputs)

# print(perceptron.prediction(np.array(inputs))) 