import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def data_visualization(X):
    fig, axis = plt.subplots(10, 10, figsize=(7, 7))
    for i in range(10):
        for j in range(10):
            axis[i, j].imshow(X[np.random.randint(0, 5001), :].reshape(20, 20, order="F"), cmap="gray")
            axis[i, j].axis("off")
    plt.show()


def main():
    data = loadmat('ex4data1.mat')
    X = data["X"]
    y = data["y"]

    # data_visualization(X)


main()
