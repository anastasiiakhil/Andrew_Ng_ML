from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualization(Y):
    plt.figure(figsize=(8, 16))
    plt.imshow(Y)
    plt.xlabel('Users')
    plt.ylabel('Movies')
    plt.show()


def cost_function_and_grad(params, Y, R, num_users, num_movies, num_features, reg_par):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)

    J = 1 / 2 * np.sum(((X @ Theta.T - Y)**2) * R)
    reg_J = J + reg_par/2 * np.sum(Theta**2) + reg_par/2 * np.sum(X**2)

    X_grad = (X @ Theta.T - Y) * R @ Theta
    Theta_grad = ((X @ Theta.T - Y) * R).T @ X
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    reg_X_grad = X_grad + reg_par*X
    reg_Theta_grad = Theta_grad + reg_par*Theta
    reg_grad = np.append(reg_X_grad.flatten(), reg_Theta_grad.flatten())
    return J, grad, reg_J, reg_grad


def normalize_ratings(Y, R):
    m, n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))
    for i in range(m):
        Ymean[i] = np.sum(Y[i, :])/np.count_nonzero(R[i, :])
        Ynorm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Ymean[i]
    return Ynorm, Ymean


def gradient_descent(initial_parameters, Y, R, num_users, num_movies, num_features, alpha, num_iters, reg_par):
    X = initial_parameters[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = initial_parameters[num_movies * num_features:].reshape(num_users, num_features)
    J_history = []

    for i in range(num_iters):
        params = np.append(X.flatten(), Theta.flatten())
        cost, grad = cost_function_and_grad(params, Y, R, num_users, num_movies, num_features,reg_par)[2:]
        X_grad = grad[:num_movies * num_features].reshape(num_movies, num_features)
        Theta_grad = grad[num_movies * num_features:].reshape(num_users, num_features)
        X = X - (alpha * X_grad)
        Theta = Theta - (alpha * Theta_grad)
        J_history.append(cost)

    params_final = np.append(X.flatten(), Theta.flatten())
    return params_final, J_history


def visualize_cost(J_history):
    plt.plot(J_history)
    plt.xlabel("Iteration")
    plt.ylabel("Theta")
    plt.title("Cost function using Gradient Descent")
    plt.show()


def main():
    movie = loadmat('ex8_movies.mat')
    params = loadmat('ex8_movieParams.mat')
    Y = movie['Y']
    R = movie['R']
    X = params['X']
    Theta = params['Theta']
    print('Average rating for movie 1 (Toy Story):', np.sum(Y[0, :] * R[0, :])/np.sum(R[0, :]), '/5\n')
    # visualization(Y)

    # To test cost function
    num_users, num_movies, num_features = 4, 5, 3
    X_test = X[:num_movies, :num_features]
    Theta_test = Theta[:num_users, :num_features]
    Y_test = Y[:num_movies, :num_users]
    R_test = R[:num_movies, :num_users]
    params = np.append(X_test.flatten(), Theta_test.flatten())
    J, grad = cost_function_and_grad(params, Y_test, R_test, num_users, num_movies, num_features, 0)[:2]
    print('Cost at loaded parameters:', J)
    J2, grad2 = cost_function_and_grad(params, Y_test, R_test, num_users, num_movies, num_features, 1.5)[2:]
    print('Cost at loaded parameters (lambda = 1.5):', J2)

    movie_list = open('movie_ids.txt', 'r', encoding='ISO-8859-1').read().split('\n')[:-1]
    my_ratings = np.zeros((1682, 1))

    # Create own ratings
    my_ratings[0] = 4
    my_ratings[69] = 5
    my_ratings[102] = 4
    my_ratings[209] = 2
    my_ratings[254] = 5
    my_ratings[282] = 5
    my_ratings[392] = 5
    my_ratings[698] = 5
    my_ratings[750] = 1
    my_ratings[868] = 4
    my_ratings[1066] = 1

    print('\nNew user ratings:')
    for i in range(len(my_ratings)):
        if my_ratings[i] > 0:
            print('Rated', int(my_ratings[i]), 'for index', movie_list[i])

    Y = np.hstack((my_ratings, Y))
    R = np.hstack((my_ratings != 0, R))
    Ynorm, Ymean = normalize_ratings(Y, R)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = 10

    X2 = np.random.randn(num_movies, num_features)
    Theta2 = np.random.randn(num_users, num_features)
    initial_parameters = np.append(X2.flatten(), Theta2.flatten())
    reg_par = 10
    params_final, J_history = gradient_descent(initial_parameters, Y, R, num_users, num_movies, num_features, 0.001, 400, reg_par)
    # visualize_cost(J_history)

    X = params_final[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params_final[num_movies * num_features:].reshape(num_users, num_features)
    p = X @ Theta.T
    my_predictions = p[:, 0][:, np.newaxis] + Ymean
    df = pd.DataFrame(np.hstack((my_predictions, np.array(movie_list)[:, np.newaxis])))
    df.sort_values(by=[0], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nTop recommendations for you:")
    for i in range(10):
        print("Predicting rating", round(float(df[0][i]), 1), " for index", df[1][i])


main()
