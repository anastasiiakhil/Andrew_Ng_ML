import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def data_visualization(X, y):
    plt.scatter(X[:, 1], y, c='red', marker='+')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of the dam')
    plt.show()


def linear_reg_cost(X, y, theta, reg_param):
    m = X.shape[0]
    theta = theta.reshape(-1, y.shape[1])
    h = np.dot(X, theta)
    cost = np.sum((h - y)**2) / (2*m) + reg_param/(2*m) * np.sum(theta[1:]**2)
    grad_0 = 1 / m * (X.transpose() @ (h-y))[0]
    grad_1 = 1 / m * (X.transpose() @ (h-y))[1:] + (reg_param / m) * theta[1:]
    grad = np.vstack((grad_0[:, np.newaxis], grad_1))
    return (cost, grad.flatten())


def train_linear_reg(X, y, reg_param):
    initial_theta = np.zeros((X.shape[1], 1))

    def cost_function(theta):
        return linear_reg_cost(X, y, theta, reg_param)

    results = minimize(fun=cost_function, x0=initial_theta, method='CG', jac=True, options={'maxiter': 200})
    new_theta = results.x
    return new_theta


def main():
    data = loadmat('ex5data1.mat')
    X, y = data['X'], data['y']
    Xval, yval  = data['Xval'], data['yval']
    Xtest, ytest = data['Xtest'], data['ytest']

    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    # data_visualization(X, y)

    theta = np.ones((X.shape[1], 1))
    cost, grad = linear_reg_cost(X, y, theta, 1)
    print('Cost and Gradient at theta = [1, 1]:',  cost, grad[0], grad[1], '(this value should be about 303.99319222026429, -15.303016, 598.250744)')

    theta_min = train_linear_reg(X, y, 0)
    print('Theta using scipy.optimize.minimize:', theta_min)


main()
