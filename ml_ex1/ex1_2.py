import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def data_normalization(x):
    return np.divide((x - np.mean(x, axis=0)), np.std(x, axis=0))


def hypothesis(x1, x2, theta):
    h = theta[0] + theta[1]*x1 + theta[2]*x2
    return h


def cost_function(x1, x2, y, theta):
    h = hypothesis(x1, x2, theta)
    J = np.sum((h - y)**2)/(2*len(y))
    return J


def gradient_descent(x1, x2, y, theta, alpha, iterations):
    J_history = []
    x1, x2, y = data_normalization(x1), data_normalization(x2), data_normalization(y)

    for i in range(iterations):
        h = hypothesis(x1, x2, theta)
        theta[0] = theta[0] - (alpha/len(y)) * sum(h-y)
        theta[1] = theta[1] - (alpha/len(y)) * sum((h-y)*x1)
        theta[2] = theta[2] - (alpha/len(y)) * sum((h-y)*x2)
        J = cost_function(x1, x2, y, theta)
        J_history.append(J)

    #  selecting learning rates
    plt.plot(range(iterations), J_history, color='g')
    plt.title('Convergence of gradient descent \n with an appropriate learning rate')
    plt.ylabel('Cost function')
    plt.xlabel('Number of iterations')
    plt.show()
    return theta


def normal_equation(x1, x2, y):
    X = np.array([np.ones(len(y)), x1, x2]).transpose()
    Y = np.array([y]).transpose()
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)
    return theta


def main():
    data = pd.read_csv('ex1data2.txt', sep=',', header=None)
    x1 = np.array(data.iloc[:, 0])
    x2 = np.array(data.iloc[:, 1])
    y = np.array(data.iloc[:, 2])
    theta = ([0, 0, 0])
    iterations = 50
    alpha = 1

    gd_theta = gradient_descent(x1, (x2), y, theta, alpha, iterations)
    ne_theta = normal_equation(data_normalization(x1), data_normalization(x2), data_normalization(y))

    print('cost function:', cost_function(x1, x2, y, gd_theta), 'gradient descent')
    print('cost function:', cost_function(x1, x2, y, ne_theta), 'normal equation')


main()
