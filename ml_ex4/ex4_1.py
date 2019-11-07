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


def sigmoid_gradient(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid * (1 - sigmoid)


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_param):
    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m, 1)), X))
    y_sort = np.zeros((m, num_labels))

    theta_1 = nn_params[:((input_layer_size+1)*hidden_layer_size)].reshape(hidden_layer_size, input_layer_size+1)
    theta_2 = nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size+1)

    # feedforward propagation
    a1 = X    # input layer
    a2 = sigmoid_function(a1 @ theta_1.T)
    a2 = np.hstack((np.ones((m, 1)), a2))   # hidden layer
    a3 = sigmoid_function(a2 @ theta_2.T)   # output layer

    # compute the unregularized and regularized cost function
    for i in range(1, num_labels + 1):
        y_sort[:, i - 1][:, np.newaxis] = np.where(y == i, 1, 0)
    for j in range(num_labels):
        J = J + sum(-y_sort[:, j] * np.log(a3[:, j]) - (1 - y_sort[:, j]) * np.log(1 - a3[:, j]))

    cost = 1/m * J
    reg_cost = cost + reg_param/(2*m) * (np.sum(theta_1[:, 1:]**2) + np.sum(theta_2[:, 1:]**2))

    # backpropagation algorithm
    grad_1 = np.zeros(theta_1.shape)
    grad_2 = np.zeros(theta_2.shape)

    for i in range(m):
        a1_i = a1[i, :]
        a2_i = a2[i, :]
        a3_i = a3[i, :]
        d3 = a3_i - y_sort[i, :]
        d2 = theta_2.T @ d3 * sigmoid_gradient(np.hstack((1, a1_i @ theta_1.T)))
        grad1 = grad_1 + d2[1:][:, np.newaxis] @ a1_i[:, np.newaxis].T
        grad2 = grad_2 + d3.T[:, np.newaxis] @ a2_i[:, np.newaxis].T

    grad_1 = 1 / m * grad_1
    grad_2 = 1 / m * grad_2

    reg_grad1 = grad_1 + (reg_param / m) * np.hstack((np.zeros((theta_1.shape[0], 1)), theta_1[:, 1:]))
    reg_grad2 = grad_2 + (reg_param / m) * np.hstack((np.zeros((theta_2.shape[0], 1)), theta_2[:, 1:]))

    return cost, grad_1, grad_2, reg_cost, reg_grad1, reg_grad2


def main():
    data = loadmat('ex4data1.mat')
    X = data["X"]
    y = data["y"]

    # data_visualization(X)

    weights = loadmat('ex4weights.mat')
    theta_1 = weights["Theta1"]
    theta_2 = weights["Theta2"]

    reg_param = 1
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    nn_params = np.append(theta_1.flatten(),theta_2.flatten())

    cost, reg_cost = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_param)[0:4:3]
    print("Cost at parameters (non-regularized):", cost, "\nCost at parameters (regularized):", reg_cost)

main()
