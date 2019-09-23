import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def visualization(data, y):
    admitted = data[y == 1]
    not_admitted = data[y == 0]
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], label='Accepted', c='navy', marker='+')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], label='Rejected', c='gold', marker='o')
    plt.xlabel('Test score 1')
    plt.ylabel('Test score 2')
    plt.legend()


def map_feature(x1, x2, degree=6):
    out = np.ones(x1.shape).reshape(len(x1),1)
    for i in range(1, degree+1):
        for j in range(i+1):
            count_feature = (x1**(i-j)*x2**j).reshape(len(x1), 1)
            out = np.hstack((out, count_feature))
    return out


def sigmoid_function(x, theta):
    z = np.dot(x, theta)
    return 1/(1+np.exp(-z))


def cost_function_reg(theta, x, y, reg_param):
    m = x.shape[0]
    h = sigmoid_function(x, theta)
    J = -sum((y * np.log(h)) + ((1 - y)*np.log(1 - h)))/m + (reg_param/m) * np.sum(theta ** 2)
    grad_0 = (np.sum((sigmoid_function(x, theta) - y)[:, None] * x, axis=0) / m)
    grad_reg = grad_0 + (reg_param / m) * theta
    grad_reg[0] = grad_0[0]
    return J


def optimize(theta, x, y, reg_param):
    return minimize(cost_function_reg, theta, args=(x, y, reg_param), tol=1e-6, options={'maxiter': 400, 'disp': True})


def visualization_boundary(data, theta, x, y):
    if x.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(x[:, 1]) - 2, np.max(x[:, 2]) + 2])
        plot_y = -(theta[0] + np.dot(theta[1], plot_x)) / theta[2]
        plt.plot(plot_x, plot_y, label='Decision Boundary', c='green')
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = np.dot(map_feature(np.array([u[i]]), np.array([v[j]])), theta)
        z = z.transpose()
        plt.contour(u, v, z, levels=[0], colors='green')
    visualization(data, y)


def main():
    data = pd.read_csv('ex2data2.txt', sep=',', header=None)
    X = np.array(data.iloc[:, 0:-1])
    y = np.array(data.iloc[:, -1])

    visualization(data, y)
    plt.title('Plot of training data')
    plt.show()

    x = map_feature(X[:, 0], X[:, 1])
    initial_theta = np.zeros(len(x[0, :]))

    res = optimize(initial_theta, x, y, 1)
    theta_1 = res.x
    visualization_boundary(data, theta_1, x, y)
    plt.title('Training data with decision boundary (λ = 1)')
    plt.show()

    res = optimize(initial_theta, x, y, 0)
    theta_2 = res.x
    visualization_boundary(data, theta_2, x, y)
    plt.title('No regularization (Overﬁtting) (λ = 0)')
    plt.show()

    res = optimize(initial_theta, x, y, 10)
    theta_3 = res.x
    visualization_boundary(data, theta_3, x, y)
    plt.title('Too much regularization (Underﬁtting) (λ = 10)')
    plt.show()

main()
