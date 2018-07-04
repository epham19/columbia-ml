import numpy as np
import sys


# Solution for Part 1
def part1(lambda_, x_train, y_train):
    dim = x_train.shape[1]
    temp = lambda_*np.eye(dim) + x_train.T.dot(x_train)
    w_rr = (np.linalg.inv(temp)).dot(x_train.T.dot(y_train))
    return w_rr


def update_posterior(lambda_, sigma2, x_train, dim, y_train, old_xx, old_xy):
    old_xx = x_train.T.dot(x_train) + old_xx
    old_xy = x_train.dot(y_train) + old_xy

    new_var_inv = lambda_ * np.eye(dim) + (1 / sigma2) * old_xx
    new_var = np.linalg.inv(new_var_inv)

    temp = lambda_ * sigma2 * np.eye(dim) + old_xx
    new_mean = (np.linalg.inv(temp)).dot(old_xy)

    return new_var, new_mean, old_xx, old_xy


# Solution for Part 2
def part2(lambda_, sigma2, x_train, y_train, x_test):
    dim = x_train.shape[1]
    active = []


    pass


def main():
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    x_train = np.genfromtxt(sys.argv[3], delimiter=",")
    y_train = np.genfromtxt(sys.argv[4])
    x_test = np.genfromtxt(sys.argv[5], delimiter=",")

    wrr = part1(lambda_input, x_train, y_train)
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wrr, delimiter="\n")  # write output to file

    active = part2(lambda_input, sigma2_input, x_train, y_train, x_test)
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active,
               delimiter=",")  # write output to file


if __name__ == "__main__":
    main()
