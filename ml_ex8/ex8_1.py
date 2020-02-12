from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def visualization_data(X):
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    plt.title('The first dataset')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()


def visualize_fit(X, mu, sigma2):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
    p2 = multivariate_Gaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
    contour_level = 10 ** np.arange(-20, 0, 3).astype(np.float)
    plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)
    plt.title('The Gaussian distribution contours of the distribution fit to the dataset')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()


def estimate_Gaussian(X):
    mu = np.sum(X, axis=0)/X.shape[0]
    var = (np.sum((X - mu) ** 2, axis=0))/X.shape[0]
    return mu, var


def multivariate_Gaussian(X, mu, sigma2):
    k = len(mu)
    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5)) * np.exp(-0.5*np.sum(X @ np.linalg.pinv(sigma2)*X, axis=1))
    return p


def main():
    data = loadmat('ex8data1.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']
    # visualization_data(X)

    mu, sigma2 = estimate_Gaussian(X)
    pval = multivariate_Gaussian(Xval, mu, sigma2)
    # visualize_fit(X, mu, sigma2)


main()
