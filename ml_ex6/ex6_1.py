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
    data_visualization(X, y, title)


def dataset_params(X, y, Xval, yval, vals):
    accuracy = 0
    best_C = 0
    best_gamma = 0
    for i in vals:
        C = i
        for j in vals:
            gamma = 1/j
            classifier = SVC(C=C, gamma=gamma)
            classifier.fit(X, y)
            prediction = classifier.predict(Xval)
            score = classifier.score(Xval, yval)
            if score > accuracy:
                accuracy = score
                best_C = C
                best_gamma = gamma
    return best_C, best_gamma


def main():
    data = loadmat('ex6data1.mat')
    X, y = data['X'], data['y']
    # data_visualization(X, y, 'Example Dataset 1')

    model1 = SVC(kernel='linear')
    model1.fit(X, np.ravel(y))
    # boundary_visualization(X, y, model1, 'SVM Decision Boundary with C = 1 (Example Dataset 1)')

    model2 = SVC(C=100, kernel='linear')
    model2.fit(X, np.ravel(y))
    # boundary_visualization(X, y, model2, 'SVM Decision Boundary with C = 100 (Example Dataset 1)')

    data2 = loadmat('ex6data2.mat')
    X2, y2 = data2['X'], data2['y']
    # data_visualization(X2, y2, 'Example Dataset 2')

    model3 = SVC(kernel="rbf", gamma=30)
    model3.fit(X2, y2.ravel())
    # boundary_visualization(X2, y2, model3, 'SVM Decision Boundary (Example Dataset 2)')

    data3 = loadmat('ex6data3.mat')
    X3, y3 = data3['X'], data3['y']
    Xval, yval = data3['Xval'], data3['yval']
    # data_visualization(X3, y3, 'Example Dataset 3')

    vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    C, gamma = dataset_params(X3, y3.ravel(), Xval, yval.ravel(), vals)
    model4 = SVC(C=C, gamma=gamma)
    model4.fit(X3, y3.ravel())
    # boundary_visualization(X3, y3, model4, 'SVM Decision Boundary (Example Dataset 3)')


main()
