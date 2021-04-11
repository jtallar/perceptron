import sys
import parser 
import json

# Read configurations from file
with open("config.json") as file:
    config = json.load(file)

training_file = config["training_file"]
expected_out = config["expected_out"]