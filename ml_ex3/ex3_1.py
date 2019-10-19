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


def lr_cost_function(theta, X, y, reg_param):

    # compute the regularized logistic cost function
    m = X.shape[0]
    h = sigmoid_function(X @ theta)
    cost = sum((-y * np.log(h)) - ((1 - y) * np.log(1 - h))) / m + reg_param / (2 * m) * sum(theta[1:] ** 2)

    # compute gradient
    grad_0 = 1 / m * (X.transpose() @ (h-y))[0]
    grad_1 = 1 / m * (X.transpose() @ (h-y))[1:] + (reg_param / m) * theta[1:]
    grad = np.vstack((grad_0[:, np.newaxis], grad_1))
    return cost[0], grad


def training_set():
    theta = np.array([-2, -1, 1, 2]).reshape(4, 1)
    X = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3, 5).T
    X = np.hstack((np.ones((5, 1)), X))
    y = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
    cost, grad = lr_cost_function(theta, X, y, 3)
    print("Cost:", cost, "Expected cost: 2.534819")
    print("Gradients:\n", grad, "\nExpected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003")


def gradient_descent(X, y, theta, alpha, iterations, reg_param):
    m = X.shape[0]
    cost_history = []

    for i in range(iterations):
        cost, grad = lr_cost_function(theta, X, y, reg_param)
        theta = theta - (alpha*grad)
        cost_history.append(cost)
    return theta, cost_history


def one_vs_all(X, y, num_labels, alpha, iterations, reg_param):
    m = X.shape[0]
    n = X.shape[1]
    initial_theta = np.zeros((n+1, 1))
    all_theta = []
    all_cost = []

    X = np.hstack((np.ones((m, 1)), X))

    for i in range(1, num_labels+1):
        theta, cost_history = gradient_descent(X, np.where(y == i, 1, 0), initial_theta, alpha, iterations, reg_param)
        all_theta.extend(theta)
        all_cost.extend(cost_history)
    return np.array(all_theta).reshape(num_labels, n+1), all_cost


def visualization_cost(all_cost, iterations):
    plt.plot(all_cost[0:iterations])
    plt.xlabel("Iteration")
    plt.ylabel("$J(\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.show()


def predict_one_vs_all(all_theta, X, y):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    predictions = X @ all_theta.T
    result = np.argmax(predictions, axis=1) + 1
    percent = sum(result[:, np.newaxis] == y)[0] / 5000 * 100
    return result, percent


def main():
    data = loadmat('ex3data1.mat')
    X = data["X"]
    y = data["y"]

    # data_visualization(X)
    # training_set()

    num_labels = 10
    alpha = 1.1
    iterations = 300
    reg_param = 1

    all_theta, all_cost = one_vs_all(X, y, num_labels, alpha, iterations, reg_param)

    # visualization_cost(all_cost, iterations)

    prediction, percent = predict_one_vs_all(all_theta, X, y)

main()
