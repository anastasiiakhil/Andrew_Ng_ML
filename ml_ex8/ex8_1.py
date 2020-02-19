from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def visualization_data(X):
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    plt.title('The first dataset')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()


def visualize_fit(X, mu, sigma2, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    X1, X2 = np.meshgrid(np.linspace(0, 35, num=70), np.linspace(0, 35, num=70))
    p2 = multivariate_Gaussian(np.hstack((X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis])), mu, sigma2)
    contour_level = 10 ** np.arange(-20, 0, 3).astype(np.float)
    plt.contour(X1, X2, p2[:, np.newaxis].reshape(X1.shape), contour_level)
    plt.title(title)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')


def visualize_anomaly(X, mu, sigma2, p, epsilon, title):
    visualize_fit(X, mu, sigma2, title)
    outliers = np.nonzero(p < epsilon)[0]
    plt.scatter(X[outliers, 0], X[outliers, 1], marker="o", facecolor="none", edgecolor="r", s=70)
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


def select_threshold(yval, pval):
    best_epsilon = 0
    best_F1 = 0
    stepsize = (max(pval) - min(pval)) / 1000
    epsilon_range = np.arange(pval.min(), pval.max(), stepsize)
    for epsilon in epsilon_range:
        predictions = (pval < epsilon)[:, np.newaxis]
        tp = np.sum(predictions[yval == 1] == 1)
        fp = np.sum(predictions[yval == 0] == 1)
        fn = np.sum(predictions[yval == 1] == 0)

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * prec * rec) / (prec + rec)
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1


def main():
    data = loadmat('ex8data1.mat')
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']
    # visualization_data(X)

    mu, sigma2 = estimate_Gaussian(X)
    p = multivariate_Gaussian(X, mu, sigma2)
    pval = multivariate_Gaussian(Xval, mu, sigma2)
    # visualize_fit(X, mu, sigma2, 'The Gaussian distribution contours of the distribution fit to the dataset')
    # plt.show()

    epsilon, F1 = select_threshold(yval, pval)
    print('Best epsilon found using cross-validation:', epsilon, '(you should see a value epsilon of about 8.99e-05)')
    print('Best F1 on Cross Validation Set:', F1, '(you should see a Best F1 value of  0.875000)\n')
    # visualize_anomaly(X, mu, sigma2, p, epsilon, 'The classified anomalies')

    # another dataset
    data2 = loadmat('ex8data2.mat')
    X2 = data2['X']
    X2val = data2['Xval']
    y2val = data2['yval']

    mu2, sigma2_2 = estimate_Gaussian(X2)
    p2 = multivariate_Gaussian(X2, mu2, sigma2_2)
    p2val = multivariate_Gaussian(X2val, mu2, sigma2_2)
    epsilon2, F1_2 = select_threshold(y2val, p2val)
    print('Best epsilon found using cross-validation:', epsilon, '(you should see a value epsilon of about 1.38e-18)')
    print('Best F1 on Cross Validation Set:', F1_2, '(you should see a Best F1 value of  0.615385)')
    print('Outliers found:', np.sum(p2 < epsilon2))


main()
