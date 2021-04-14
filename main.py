import sys
import parser 
import json
import numpy as np
import simple

# Read configurations from file
with open("config.json") as file:
    config = json.load(file)

# training_file = config["training_file"]
# expected_out = config["expected_out"]
max_iter = config["max_iter"]
eta = config["eta"]

inputs = [[1,1], [1,0], [0,1], [0,0]]
outputs = np.array([1, 0, 0, 1])

perceptron = simple.SimplePerceptron(max_iter, eta)
perceptron.train(np.array(inputs), outputs)

inputs = np.array([[1,1], [1,0], [0,1], [0,0]])
print(perceptron.prediction(inputs)) 