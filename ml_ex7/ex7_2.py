from scipy.io import loadmat
import matplotlib.pyplot as plt


def visualization_data(X):
    plt.scatter(X[:, 0], X[:, 1], marker="o", facecolors="none", edgecolors="b")
    plt.title('Example Dataset 1')
    plt.show()


def main():
    data = loadmat('ex7data1.mat')
    X = data["X"]
    # visualization_data(X)


main()
