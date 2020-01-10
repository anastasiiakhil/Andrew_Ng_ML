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
        title = "Iteration Number " + str(i+1)
        plt.title(title)

        centroids = compute_centroids(X, idx, K)
        idx = find_closest_centroids(X, centroids)
        plt.tight_layout()
        plt.pause(5e-1)
    plt.show()


def kmeans_init_centroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    for i in range(K):
        centroids[i] = X[np.random.randint(0, X.shape[0] + 1), :]
    return centroids


def run_kmeans(X, initial_centroids, num_iters, K):
    idx = find_closest_centroids(X, initial_centroids)
    for i in range(num_iters):
        centroids = compute_centroids(X, idx, K)
        idx = find_closest_centroids(X, initial_centroids)
        return centroids, idx


def display_images(X, X_recovered, size):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(X.reshape(size[0], size[1], size[2]))
    ax[1].imshow(X_recovered.reshape(size[0], size[1], size[2]))
    fig.suptitle('Original and reconstructed image')
    plt.show()


def main():
    data = loadmat('ex7data2.mat')
    X = data["X"]

    # select an initial set of centroids
    example_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    example_idx = find_closest_centroids(X, example_centroids)
    # print("Closest centroids for the first 3 examples:\n", example_idx[0:3])
    # print('(the closest centroids should be 1, 3, 2 respectively)\n')

    K = 3
    centroids = compute_centroids(X, example_idx, K)
    # print("Centroids computed after initial finding of closest centroids:\n", centroids)
    # visualization_kmeans(X, example_centroids, example_idx, K, 10)

    init_centroids = kmeans_init_centroids(X, K)
    init_idx = find_closest_centroids(X, init_centroids)
    # visualization_kmeans(X, init_centroids, init_idx, K, 10)

    data2 = loadmat("bird_small.mat")
    A = data2["A"]
    size = A.shape
    X2 = (A/255).reshape(size[0]*size[1], size[2])

    K2 = 16
    initial_centroids2 = kmeans_init_centroids(X2, K2)
    centroids2, idx2 = run_kmeans(X2, initial_centroids2, 10, K2)
    X2_recovered = X2.copy()
    for i in range(1, K2 + 1):
        X2_recovered[(idx2 == i).ravel(), :] = centroids2[i - 1]
    # display_images(X2, X2_recovered, size)


main()
