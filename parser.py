import numpy as np

def training(train_name, out_name):
    train_file = open(train_name, "r")
    output = outputs(out_name)
    data = []
    final = []
    for line in train_file:
        data = []
        data.append(1.0)
        for n in line.split():
            data.append(float(n))
        final.append(data)
    return np.array(final), np.array(output)

def outputs(file_name):
    output_file = open(file_name, "r")
    outputs = []
    for line in output_file:
        output = float(line.split()[0])
        outputs.append(output)
    return outputs