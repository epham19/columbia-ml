from __future__ import division
import numpy as np
import sys


def plugin_classifier(x_train, y_train, x_test):
    pass


def main():
    x_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    x_test = np.genfromtxt(sys.argv[3], delimiter=",")

    final_outputs = plugin_classifier(x_train, y_train, x_test)

    # write output to file
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")
