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
    old_xy = x_train.T.dot(y_train) + old_xy

    new_var_inv = lambda_ * np.eye(dim) + (1 / sigma2) * old_xx
    new_var = np.linalg.inv(new_var_inv)

    temp = lambda_ * sigma2 * np.eye(dim) + old_xx
    new_mean = (np.linalg.inv(temp)).dot(old_xy)

    return new_var, new_mean, old_xx, old_xy


# Solution for Part 2
def part2(lambda_, sigma2, x_train, y_train, x_test):
    dim = x_train.shape[1]
    active = []

    old_xx = np.zeros((dim, dim))
    old_xy = np.zeros(dim)

    new_var, new_mean, old_xx, old_xy = update_posterior(lambda_, sigma2, x_train, dim, y_train, old_xx, old_xy)

    w_rr = new_mean

    # Create 1-based indexes
    indices = list(range(x_test.shape[0]))

    # Select 10 data points to measure
    for i in range(0, 10):
        # Pick x_0 for which sigma2_0 is largest
        var_matrix = (x_test.dot(new_var)).dot(x_test.T)
        row_largest = np.argmax(var_matrix.diagonal())

        # Update x and y values
        x_train = x_test[row_largest, :]
        y_train = x_train.dot(w_rr)

        actual_row = indices[row_largest]
        active.append(actual_row)

        # Remove x_0
        x_test = np.delete(x_test, row_largest, axis=0)
        indices.pop(row_largest)

        # Update posterior distribution
        new_var, new_mean, old_xx, old_xy = update_posterior(lambda_, sigma2, x_train, dim, y_train, old_xx, old_xy)

        w_rr = new_mean

    # Create 1-based indexes
    active = [i + 1 for i in active]
    return active


def main():
    lambda_input = int(sys.argv[1])
    sigma2_input = float(sys.argv[2])
    x_train = np.genfromtxt(sys.argv[3], delimiter=",")
    y_train = np.genfromtxt(sys.argv[4])
    x_test = np.genfromtxt(sys.argv[5], delimiter=",")

    # write output to file
    wrr = part1(lambda_input, x_train, y_train)
    np.savetxt("wRR_" + str(lambda_input) + ".csv", wrr, delimiter="\n")

    active = part2(lambda_input, sigma2_input, x_train, y_train, x_test)
    np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", np.array(active).reshape(1, np.array(active).shape[0]), delimiter=",", fmt='%d')


if __name__ == "__main__":
    main()
