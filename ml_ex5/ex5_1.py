import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def data_visualization(X, y):
    plt.scatter(X, y, c='red', marker='x')
    plt.title('Data')
    plt.xlabel('Change in water level')
    plt.ylabel('Water flowing out of the dam')
    plt.show()


def data_with_line(X, y, line):
    plt.scatter(X, y, c='red', marker='x')
    plt.plot(X, line)
    plt.title('Linear Fit')
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


def learning_curve(X, y, Xval, yval, reg_param):
    m = X.shape[0]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(1, m+1):
        theta = train_linear_reg(X[:i], y[:i], reg_param)
        error_train[i-1] = linear_reg_cost(X[:i], y[:i], theta, 0)[0]
        error_val[i-1] = linear_reg_cost(Xval, yval, theta, 0)[0]
    return error_train, error_val


def curve_visualization(X, error_train, error_val):
    m = X.shape[0]
    plt.plot(range(1, m+1), error_train, c='blue', label='Train')
    plt.plot(range(1, m+1), error_val, c='green', label='Cross Validation')
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


def poly_features(X, p):
    poly_X = np.zeros((X.shape[0], p))
    for i in range(p):
        poly_X[:, i] = X[:, 0]**(i+1)
    return poly_X


def data_normalization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std((x - mu), axis=0)
    X_norm = (x - mu)/sigma
    return X_norm, mu, sigma


def main():
    data = loadmat('ex5data1.mat')
    X, y = data['X'], data['y']
    Xval, yval  = data['Xval'], data['yval']
    Xtest, ytest = data['Xtest'], data['ytest']
    X_with_1s = np.insert(X, 0, 1, axis=1)
    Xval_with_1s = np.insert(Xval, 0, 1, axis=1)
    # data_visualization(X, y)

    theta = np.array([[1], [1]])
    cost, grad = linear_reg_cost(X_with_1s, y, theta, 1)
    print('Cost and Gradient at theta = [1, 1]:',  cost, grad[0], grad[1], '(this value should be about 303.99319222026429, -15.303016, 598.250744)')

    theta_min = train_linear_reg(X_with_1s, y, 0)
    print('Theta using scipy.optimize.minimize:', theta_min)
    # data_with_line(X, y, np.dot(X_with_1s, theta_min))

    error_train, error_val = learning_curve(X_with_1s, y, Xval_with_1s, yval, 0)
    # curve_visualization(X, error_train, error_val)

    # Feature Mapping for Polynomial Regression
    p = 8
    poly_X = poly_features(X, p)
    poly_X, mu, sigma = data_normalization(poly_X)
    poly_X = np.hstack((np.ones((poly_X.shape[0], 1)), poly_X))

    poly_test_X = poly_features(Xtest, p)
    poly_test_X, mu, sigma = data_normalization(poly_test_X)
    poly_test_X = np.hstack((np.ones((poly_test_X.shape[0], 1)), poly_test_X))

    poly_val_X = poly_features(Xval, p)
    poly_val_X = poly_val_X - mu
    poly_val = poly_val_X / sigma
    poly_val_X = np.hstack((np.ones((poly_val_X.shape[0], 1)), poly_val_X))


main()
