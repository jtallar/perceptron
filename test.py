import json

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import statistics as sts

import main


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        s = self.fmt % x
        dec_point = '.'
        pos_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(dec_point)
        sign = tup[1][0].replace(pos_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if not exponent: exponent = 0
        exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s = r'%s{\times}%s' % (significand, exponent)
        else:
            s = r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def plot_error_bars(x_values, x_label, y_values, y_label, y_error):
    fig, ax = plt.subplots(figsize=(12, 10))  # Create a figure containing a single axes.
    (_, caps, _) = plt.errorbar(x_values, y_values, yerr=y_error, markersize=6, capsize=20, elinewidth=0.75,
                                linestyle='-', marker='o')  # Plot some data on the axes
    for cap in caps:
        cap.set_markeredgewidth(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


count: int = 10
input_range: range = np.arange(10, 100, 10)
with open("config.json") as file:
    config = json.load(file)

final_accuracy_test_mean = []
final_accuracy_test_std = []

final_accuracy_train_mean = []
final_accuracy_train_std = []

for ratio in input_range:
    accuracy_train = []
    accuracy_test = []

    for i in range(count):
        aux_dict: dict = main.func(config, ratio)
        accuracy_train.append(aux_dict['acc_train'])
        accuracy_test.append(aux_dict['acc_test'])
        print(f"ratio: {ratio}, index: {i}")

    final_accuracy_train_mean.append(sts.mean(accuracy_train))
    final_accuracy_train_std.append(sts.stdev(accuracy_train))
    final_accuracy_test_mean.append(sts.mean(accuracy_test))
    final_accuracy_test_std.append(sts.stdev(accuracy_test))

plot_error_bars(np.array(reversed(input_range)), 'Test Set (%)', final_accuracy_test_mean, 'Testing Accuracy',
                final_accuracy_test_std)

plot_error_bars(input_range, 'Train Set (%)', final_accuracy_train_mean, 'Training Accuracy',
                final_accuracy_train_std)

