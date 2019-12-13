from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np


def data_visualization(X, y, title):
    m, n = X.shape[0], X.shape[1]
    pos = (y == 1).reshape(m, 1)
    neg = (y == 0).reshape(m, 1)
    plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c='r', marker='+', s=50)
    plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], facecolors='yellow', edgecolors='black', s=50)
    plt.title(title)
    plt.show()


def boundary_visualization(X, y, model, title):
    X_1, X_2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 1].max(), num=100), np.linspace(X[:, 1].min(), X[:, 1].max(), num=100))
    plt.contour(X_1, X_2, model.predict(np.array([X_1.ravel(), X_2.ravel()]).T).reshape(X_1.shape), 1, colors="b")
    plt.xlim(0, 4.5)
    plt.ylim(1.5, 5)
    data_visualization(X, y, title)


def main():
    data = loadmat('ex6data1.mat')
    X, y = data['X'], data['y']

    # data_visualization(X, y, 'Example Dataset 1')

    model = SVC(kernel='linear')
    model.fit(X, np.ravel(y))
    # boundary_visualization(X, y, model, 'SVM Decision Boundary with C = 1 (Example Dataset 1)')

    model = SVC(C=100, kernel='linear')
    model.fit(X, np.ravel(y))
    # boundary_visualization(X, y, model, 'SVM Decision Boundary with C = 100 (Example Dataset 1)')


main()
