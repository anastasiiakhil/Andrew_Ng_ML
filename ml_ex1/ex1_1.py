import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab


def hypothesis(x, theta):
    h = theta[0] + theta[1]*x
    return h


def cost_function(x, y, theta):
    h = hypothesis(x, theta)
    J = np.sum((h - y)**2)/(2*len(y))
    return J


def gradient_descent(x, y, theta, alpha, iterations):
    for i in range(iterations):
        h = hypothesis(x, theta)
        theta[0] = theta[0] - (alpha/len(y)) * sum(h - y)
        theta[1] = theta[1] - (alpha/len(y)) * sum((h-y)*x)
        visual_animation(x, y, h, 'Gradient Descent')
    return theta


def normal_equation(x, y):
    X = np.array([np.ones(len(y)), x]).transpose()
    Y = np.array([y]).transpose()
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)
    return theta


def visualization(x, y, line, title):
    plt.figure()
    plt.title(title)
    plt.scatter(x, y, c='red', marker='+')
    plt.plot(x, line, color='g')
    plt.show()


def visual_animation(x, y, h, title):
    pylab.clf()
    plt.title(title)
    pylab.scatter(x, y, c='red', marker='+')
    plt.plot(x, h, color='g')
    plt.pause(5e-100)


def main():
    data = pd.read_csv('ex1data1.txt', sep=',', header=None)
    x = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])

    iterations = 1500
    alpha = 0.01
    theta = list([0, 0])


    line = hypothesis(x, theta)
    print('cost function:', cost_function(x, y, theta), 'first step')
    visualization(x, y, line, "Data")

    G = gradient_descent(x, y, theta, alpha, iterations)
    print('cost function:', cost_function(x, y, G), 'gradient descent')


    NE = normal_equation(x, y)
    print('cost function:', cost_function(x, y, NE), 'normal equation')
    visualization(x, y, hypothesis(x, NE), "Normal Equation")


main()
