from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd


def visualization_data(X):
    plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none", edgecolors="b")
    plt.title('Example Dataset 1')
    plt.show()


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def pca(X):
    m, n = X.shape[0], X.shape[1]
    sigma = 1/m * X.T @ X
    U, S, V = svd(sigma)
    return U, S, V


def visualization_eigenvector(X, mu, U, S):
    plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none", edgecolors="b")
    plt.plot([mu[0], (mu + 1.5 * S[0] * U[:, 0].T)[0]], [mu[1], (mu + 1.5 * S[0] * U[:, 0].T)[1]], color="black", linewidth=3)
    plt.plot([mu[0], (mu + 1.5 * S[1] * U[:, 1].T)[0]], [mu[1], (mu + 1.5 * S[1] * U[:, 1].T)[1]], color="black", linewidth=3)
    plt.xlim(-1, 7)
    plt.ylim(2, 8)
    plt.title('Computed eigenvectors of the dataset')
    plt.show()


def project_data(X, U, K):
    Z = np.zeros((X.shape[0], K))
    U_reduce = U[:, :K]
    for i in range(X.shape[0]):
        for j in range(K):
            Z[i, j] = X[i, :] @ U_reduce[:, j]
    return Z


def recover_data(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_reduce = U[:, :K]
    for i in range(Z.shape[0]):
        X_rec[i, :] = Z[i, :] @ U_reduce.T
    return X_rec


def visualization_after_PCA(X_norm, X_rec):
    plt.scatter(X_norm[:, 0], X_norm[:, 1], marker="o", label="Original", facecolors="none", edgecolors="b")
    plt.scatter(X_rec[:, 0], X_rec[:, 1], marker="o", label="Approximation", facecolors="none", edgecolors="r")
    plt.title("The Normalized and Projected Data after PCA")
    plt.legend()
    plt.show()


def main():
    data = loadmat('ex7data1.mat')
    X = data["X"]
    # visualization_data(X)

    X_norm, mu, sigma = feature_normalize(X)
    U, S, V = pca(X_norm)
    # visualization_eigenvector(X, mu, U, S)

    K = 1
    Z = project_data(X_norm, U, K)
    print("Projection of the first example:", Z[0][0], '\n(this value should be about 1.481274)\n')

    X_rec = recover_data(Z, U, K)
    print("Approximation of the first example:", X_rec[0, :], '\n(this value should be about  -1.047419 -1.047419)\n')
    # visualization_after_PCA(X_norm, X_rec)


main()
