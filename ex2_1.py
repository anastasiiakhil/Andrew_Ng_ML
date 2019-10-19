import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_tnc


def visualization(data, y):
    admitted = data[y == 1]
    not_admitted = data[y == 0]
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], label='Admitted', c='navy', marker='+')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], label='Not Admitted', c='gold', marker='o')
    plt.xlabel('Exam 1 ')
    plt.ylabel('Exam 2 ')
    plt.title('Scatter plot of training data')
    plt.legend()


def sigmoid_function(x, theta):
    z = np.dot(x, theta)
    return 1/(1+np.exp(-z))


def cost_function(theta, x, y):
    m = x.shape[0]
    J = -(1/m) * np.sum(y*np.log(sigmoid_function(x, theta))+(1-y)*np.log(1-sigmoid_function(x, theta)))
    return J


def gradient_descent(theta, x, y):
    m = x.shape[0]
    return (1 / m) * np.dot(x.transpose(), sigmoid_function(x, theta) - y)


def minimizes(x, y, theta):
    weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient_descent, args=(x, y.flatten()))
    return weights[0]


def visualization_boundary(data, x, y, theta):
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = [np.min(x[:, 1])-2, np.max(x[:, 1])+2]
    plot_y = - (theta[0] + np.dot(theta[1], plot_x)) / theta[2]
    plt.plot(plot_x, plot_y, label='Decision Boundary', c='green')
    visualization(data, y)
    plt.title('Training data with decision boundary')


def main():
    data = pd.read_csv('ex2data1.txt', sep=',', header=None)
    X = data.iloc[:, :-1]
    Y = np.array(data.iloc[:, -1])
    x = np.c_[np.ones((X.shape[0], 1)), X]
    y = Y[:, np.newaxis]
    theta = np.zeros((x.shape[1], 1))

    visualization(data, y)
    plt.show()

    sigmoid = sigmoid_function(x, theta)
    cost = cost_function(theta, x, y)
    new_theta_parameters = minimizes(x, y, theta)

    # For example let`s predict admission probability when 1 exam = 45 and 2 exam = 85"
    predict = sigmoid_function([1, 45, 85], new_theta_parameters)

    p = np.round(sigmoid_function(x, new_theta_parameters))
    predictions = np.zeros(p.shape)
    predictions[p >= 0.5] = 1

    visualization_boundary(data, x, y, new_theta_parameters)
    plt.show()
    print('new theta parameters:', new_theta_parameters)

main()
