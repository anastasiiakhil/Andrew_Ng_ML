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


def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))


def predict(theta_1, theta_2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    a_1 = sigmoid_function(X @ np.transpose(theta_1))

    # hidden layer
    a_1 = np.hstack((np.ones((m, 1)), a_1))

    # output layer
    a_2 = sigmoid_function(a_1 @ np.transpose(theta_2))
    return np.argmax(a_2, axis=1) + 1


def display_data(k):
    plt.imshow(k, cmap="gray")
    plt.show()


def randomly(X, theta_1, theta_2):
    m = X.shape[0]
    rp = np.random.permutation(m)

    for i in range(m):
        print('Displaying Example Image')
        image = np.array([X[rp[i]]])
        display_data(image.reshape(20, 20, order="F"))

        pred = predict(theta_1, theta_2, image)
        print('Neural Network Prediction:', pred % 10)
        input('Program paused. Press enter to continue')


def main():
    data = loadmat('ex3data1.mat')
    X = data['X']
    y = data['y']

    # data_visualization(X)

    weights = loadmat('ex3weights.mat')
    theta_1 = weights["Theta1"]
    theta_2 = weights["Theta2"]

    prediction = predict(theta_1, theta_2, X)
    print('Training Set Accuracy:', sum(prediction[:, np.newaxis] == y)[0] / 5000 * 100, "%")

    # randomly(X, theta_1, theta_2)


main()
