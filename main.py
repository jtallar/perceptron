import sys
import parser 
import json
import numpy as np
import simple

# Read configurations from file
with open("config.json") as file:
    config = json.load(file)

max_iter = config["max_iter"]
eta = config["eta"]
training_linear = config["training_file"]
out_linear = config["expected_out"]

inputs = [[1,1], [1,0], [0,1], [0,0]]
outputs = np.array([1, 0, 0, 0])

perceptron = simple.SimplePerceptron(max_iter, eta)
perceptron.train(np.array(inputs), outputs)

inputs = np.array([[1,1], [1,0], [0,1], [0,0]])
print(perceptron.prediction(inputs)) 