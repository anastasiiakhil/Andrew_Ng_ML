import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def data_visualization(X, y, title, xlabel, ylabel):
    plt.scatter(X, y, c='red', marker='x')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def data_with_line(X, y, line, title, xlabel, ylabel):
    plt.plot(X, line)
    data_visualization(X, y, title, xlabel, ylabel)


def linear_reg_cost(X, y, theta, reg_param):
    m = X.shape[0]
    theta = theta.reshape(-1, y.shape[1])
    h = np.dot(X, theta)
    cost = 1/(2*m) * np.sum((h - y)**2) + (reg_param / (2 * m)) * np.sum((theta[1:])**2)
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


def learning_curve(X, y, Xval, yval, reg_param):
    m = X.shape[0]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(1, m+1):
        theta = train_linear_reg(X[:i], y[:i], reg_param)
        error_train[i-1] = linear_reg_cost(X[:i], y[:i], theta, 0)[0]
        error_val[i-1] = linear_reg_cost(Xval, yval, theta, 0)[0]
    return error_train, error_val


def curve_visualization(X, error_train, error_val, title):
    m = X.shape[0]
    plt.plot(range(1, m+1), error_train, c='blue', label='Train')
    plt.plot(range(1, m+1), error_val, c='green', label='Cross Validation')
    plt.title(title)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


def poly_features(X, p):
    poly_X = X
    for i in range(1, p):
        poly_X = np.column_stack((poly_X, np.power(X, i + 1)))
    return poly_X


def data_normalization(x):
    mu = np.mean(x, axis=0)
    x_norm = x - mu
    sigma = np.std(x_norm, axis=0)
    x_norm = x_norm / sigma
    return x_norm, mu, sigma


def plot_fit(X, y, title, xlabel, ylabel, mu, sigma, theta, p):
    x = np.array(np.arange(min(X) - 15, max(X) + 25, 0.05))
    poly_X = poly_features(x, p)
    poly_X = (poly_X - mu) / sigma
    poly_X = np.hstack((np.ones((poly_X.shape[0], 1)), poly_X))
    plt.plot(x, np.dot(poly_X, theta))
    data_visualization(X, y, title, xlabel, ylabel)


def main():
    data = loadmat('ex5data1.mat')
    X, y = data['X'], data['y']
    Xval, yval  = data['Xval'], data['yval']
    Xtest, ytest = data['Xtest'], data['ytest']
    X_with_1s = np.insert(X, 0, 1, axis=1)
    Xval_with_1s = np.insert(Xval, 0, 1, axis=1)
    xlabel = 'Change in water level'
    ylabel = 'Water flowing out of the dam'
    # data_visualization(X, y, 'Data', xlabel, ylabel)

    theta = np.array([[1], [1]])
    cost, grad = linear_reg_cost(X_with_1s, y, theta, 1)
    print('Cost and Gradient at theta = [1, 1]:',  cost, grad[0], grad[1], '(this value should be about 303.99319222026429, -15.303016, 598.250744)')

    theta_min = train_linear_reg(X_with_1s, y, 0)
    print('Theta using scipy.optimize.minimize:', theta_min)
    # data_with_line(X, y, np.dot(X_with_1s, theta_min), 'Linear Fit', xlabel, ylabel)

    error_train, error_val = learning_curve(X_with_1s, y, Xval_with_1s, yval, 0)
    # curve_visualization(X, error_train, error_val, 'Learning curve for linear regression')

    # Feature Mapping for Polynomial Regression
    p = 8
    poly_X = poly_features(X, p)
    poly_X, mu, sigma = data_normalization(poly_X)
    poly_X = np.hstack((np.ones((poly_X.shape[0], 1)), poly_X))

    poly_test_X = poly_features(Xtest, p)
    poly_test_X = (poly_test_X - mu) / sigma
    poly_test_X = np.hstack((np.ones((poly_test_X.shape[0], 1)), poly_test_X))

    poly_val_X = poly_features(Xval, p)
    poly_val_X = (poly_val_X - mu) / sigma
    poly_val_X = np.hstack((np.ones((poly_val_X.shape[0], 1)), poly_val_X))

    poly_theta = train_linear_reg(poly_X, y, 0)
    # plot_fit(X, y, 'Polynomial fit, $\lambda$ = 0', xlabel, ylabel, mu, sigma, poly_theta, p)

    error_train_poly, error_val_poly = learning_curve(poly_X, y, poly_val_X, yval, 0)
    # curve_visualization(X, error_train_poly, error_val_poly, 'Polynomial learning curve, $\lambda$ = 0')


main()
