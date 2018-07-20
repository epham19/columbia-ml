# This script implements a K-class Bayes classifier to compute the probability
# of a new data point belonging to each of the K-class using the Gaussian class conditional densities

from __future__ import division
import numpy as np
import sys


def class_prior(x_train, y_train):
    """Derive the maximum likelihood updates for the class prior probability vector"""
    length = x_train.shape[0]
    class_label, count = np.unique(y_train, return_counts=True)

    # Compute MLE
    class_prob = (count / float(length)).T
    return class_prob


def class_cond_density(x_train, y_train, num_classes):
    """Compute the class-specific Gaussian mean and covariance"""
    dim = x_train.shape[1]

    # Initialise covariance and means
    cov = np.zeros((dim, dim, num_classes))
    mean = np.zeros((num_classes, dim))

    # Compute empirical MLE mean and covariance of class y
    for i in range(num_classes):
        x_i = x_train[(y_train == i)]
        mean[i] = np.mean(x_i, axis=0)

        x_i_n = x_i - mean[i]
        temp = (x_i_n.T).dot(x_i_n)
        cov[:, :, i] = temp / float(len(x_i))

    return mean, cov


def plugin_classifier(x_test, class_prob, mean, cov, num_classes):
    length = x_test.shape[0]
    prob = np.zeros((length, num_classes))
    prob_norm = np.zeros((length, num_classes))

    for k in range(num_classes):
        inv_cov = np.linalg.inv(cov[:, :, k])
        inv_sqr_det_cov = (np.linalg.det(cov[:, :, k])) ** -0.5
        for i in range(length):
            x_0 = x_test[i, :]
            temp = ((x_0 - mean[k]).T).dot(inv_cov).dot(x_0 - mean[k])
            prob[i, k] = class_prob[k] * inv_sqr_det_cov * np.exp(-0.5 * temp)

    for i in range(length):
        total = prob[i, :].sum()
        prob_norm[i, :] = prob[i, :] / float(total)

    return prob_norm


def main():
    x_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    x_test = np.genfromtxt(sys.argv[3], delimiter=",")

    num_classes = 10

    class_prob = class_prior(x_train, y_train)

    mean, cov = class_cond_density(x_test, y_train, num_classes)

    final_outputs = plugin_classifier(x_train, class_prob, mean, cov, num_classes)

    # write output to file
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")


if __name__ == "__main__":
    main()
