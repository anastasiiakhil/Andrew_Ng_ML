import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pylab


def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0], 1))
    temp = np.zeros((centroids.shape[0], 1))

    for i in range(X.shape[0]):
        for j in range(K):
            dist = X[i, :] - centroids[j, :]
            length = np.sum(dist ** 2)
            temp[j] = length
        idx[i] = np.argmin(temp) + 1
    return idx


def compute_centroids(X, idx, K):
    centroids = np.zeros((K, X.shape[1]))
    count = np.zeros((K, 1))

    for i in range(X.shape[0]):
        index = int((idx[i] - 1)[0])
        centroids[index, :] += X[i, :]
        count[index] += 1
    return centroids/count


def visualization_kmeans(X, centroids, idx, K, num_iters):
    for i in range(num_iters):
        pylab.clf()
        color = "rgb"

        for k in range(1, K + 1):
            grp = (idx == k).reshape(X.shape[0], 1)
            plt.scatter(X[grp[:, 0], 0], X[grp[:, 0], 1], c=color[k - 1], s=15)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x", c="black", linewidth=3)
        title = "Iteration Number " + str(i)
        plt.title(title)

        centroids = compute_centroids(X, idx, K)
        idx = find_closest_centroids(X, centroids)
        plt.tight_layout()
        plt.pause(5e-1)
    plt.show()


def main():
    data = loadmat('ex7data2.mat')
    X = data["X"]

    # select an initial set of centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroids(X, initial_centroids)
    # print("Closest centroids for the first 3 examples:\n", idx[0:3])
    # print('(the closest centroids should be 1, 3, 2 respectively)\n')

    K = 3
    centroids = compute_centroids(X, idx, K)
    # print("Centroids computed after initial finding of closest centroids:\n", centroids)

    # visualization_kmeans(X, initial_centroids, idx, K, 10)


main()
