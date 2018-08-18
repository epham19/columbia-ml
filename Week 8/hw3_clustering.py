from __future__ import division

import numpy as np
import pandas as pd
import scipy as sp
import sys

def kmeans(x_data):
    # Perform the k-means clustering algorithm with 5 clusters and 10 iterations
    num_clusters = 5
    num_iteration = 10
    length = x_data.shape[0]

    # Create cluster assignment vector
    c = np.zeros(length)

    # Initialize mu with a uniform random selection of data points
    indices = np.random.randint(0, length, size=num_clusters)
    mu = x_data[indices]

    for i in range(num_iteration):
        # Update cluster assignments c_i
        for i_, x_i in enumerate(x_data):
            temp1 = np.linalg.norm(mu - x_i, 2, 1)
            c[i_] = np.argmin(temp1)

        # Update cluster mu
        n = np.bincount(c.astype(np.int64), None, num_clusters)
        for k in range(num_clusters):
            indices = np.where(c == k)[0]
            mu[k] = (np.sum(x_data[indices], 0)) / float(n[k])

        # Write output to file
        filename = "centroids-" + str(i+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")


def emgmm(x_data):
    # Perform the EM algorithm to a Gaussian mixture model with 5 clusters and 10 iterations
    num_classes = 5
    num_iteration = 10
    length = x_data.shape[0]
    dim = x_data.shape[1]
    sigma_k = np.eye(dim)
    sigma = np.repeat(sigma_k[:, :, np.newaxis], num_classes, axis=2)
    pi_class = np.ones(num_classes) * (1 / num_classes)
    phi = np.zeros((length, num_classes))
    phi_norm = np.zeros((length, num_classes))
    indices = np.random.randint(0, length, size=num_classes)

    # Initialize mu with a uniform random selection of data points
    mu = x_data[indices]

    for i in range(num_iteration):
        # Compute E-step of EM algorithm
        for k in range(num_classes):
            inv_sigma_k = np.linalg.inv(sigma[:, :, k])
            inv_sqr_det_sigma_k = (np.linalg.det(sigma[:, :, k])) ** -0.5
            for index in range(length):
                x_i = x_data[index, :]
                temp1 = (((x_i - mu[k]).T).dot(inv_sigma_k)).dot(x_i - mu[k])
                phi[index, k] = pi_class[k] * ((2 * np.pi) ** (-dim / 2)) * inv_sqr_det_sigma_k * np.exp(-0.5 * temp1)
            for index in range(length):
                tot = phi[index, :].sum()
                phi_norm[index, :] = phi[index, :] / float(tot)

        # Compute M-step of the EM algorithm
        n_k = np.sum(phi_norm,axis=0)
        pi_class = n_k/float(length)
        for k in range(num_classes):
            mu[k] = ((phi_norm[:,k].T).dot(x_data))/n_k[k]

        for k in range(num_classes):
            temp1 = np.zeros((dim,1))
            temp2 = np.zeros((dim,dim))
            for index in range(length):
                xi = x_data[index,:]
                temp1[:,0] = xi - mu[k]
                temp2 = temp2 + phi_norm[index,k]*np.outer(temp1,temp1)
            sigma[:,:,k] = temp2/float(n_k[k])

        # Write output to file
        filename = "pi-" + str(i+1) + ".csv"
        np.savetxt(filename, pi_class, delimiter=",")

        filename = "mu-" + str(i+1) + ".csv"
        np.savetxt(filename, mu, delimiter=",")

        for j in range(num_classes):
            filename = "Sigma-" + str(j+1) + "-" + str(i+1) + ".csv"
            np.savetxt(filename, sigma[:,:,j], delimiter=",")


def main():
    x_data = np.genfromtxt(sys.argv[1], delimiter=",")
    kmeans(x_data)
    emgmm(x_data)


if __name__ == "__main__":
    main()
