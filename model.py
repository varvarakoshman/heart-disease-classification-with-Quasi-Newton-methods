import numpy as np
import pandas as pd
import scipy.optimize as sp
from sklearn.model_selection import train_test_split

from util import plot_cost_history

# dataset and learning rate are available globally
lr = 0.01
epsilon = 0.0001
data = pd.read_csv("../resources/heart.csv")
y_data = data.target.values
x_data = data.drop(['target'], axis=1)
x_data_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x_data_norm, y_data, test_size=0.2, random_state=0)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon
cost_function = lambda theta: np.mean(-(y_train * np.log(sigmoid(np.dot(theta.T, x_train))) + (1 - y_train) * np.log(
    1 - sigmoid(np.dot(theta.T, x_train)))))
jac = lambda point: np.dot(x_train, (sigmoid(np.dot(point.T, x_train)) - y_train).T) / x_train.shape[1]


def main():
    parameters_gd, iterations_gd = gradient_descent()
    y_prediction = predict(np.array([q[0] for q in parameters_gd]))
    print("Test Accuracy for GD: {:.2f}% with {} iterations".format(
        (100 - np.mean(np.abs(y_prediction - y_test)) * 100), iterations_gd))
    optimal_newton, newton_iterations = newton_method()
    y_prediction = predict(optimal_newton)
    print("Test Accuracy for Newton method: {:.2f}% with {} iterations".format(
        (100 - np.mean(np.abs(y_prediction - y_test)) * 100), newton_iterations))
    optimal_bfgs, bfgs_iterations = bfgs_method()
    y_prediction = predict(optimal_bfgs)
    print("Test Accuracy for BFGS method: {:.2f}% with {} iterations".format(
        (100 - np.mean(np.abs(y_prediction - y_test)) * 100), bfgs_iterations))
    optimal_lbfgs, lbfgs_iterations = lbfgs_method()
    y_prediction = predict(optimal_lbfgs)
    print("Test Accuracy for Limited memory BFGS method: {:.2f}% with {} iterations".format(
        (100 - np.mean(np.abs(y_prediction - y_test)) * 100), lbfgs_iterations))


def costF(param):
    h = sigmoid(np.dot(param.T, x_train))
    loss = -(y_train * np.log(h) + (1 - y_train) * np.log(1 - h))
    cost = np.mean(loss)
    gradient = np.dot(x_train, (h - y_train).T) / x_train.shape[1]
    return cost, gradient


def sigmoid(z):
    result = 1 / (1 + np.exp(-z))
    return result


def predict(param):
    h = sigmoid(np.dot(param.T, x_test))
    y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(len(h)):
        if h[i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1
    return y_prediction


def gradient_descent():
    prev = np.random.rand(x_train.shape[0], 1)
    iterations = 0
    cost_history = []
    while True:
        cost, gradient = costF(prev)
        curr = prev - lr * gradient
        iterations += 1
        cost_history.append(cost)
        if stop_condition(prev, curr):
            break
        else:
            prev = curr
    plot_cost_history(iterations, cost_history)
    return curr, iterations


def newton_method():
    newton_res = sp.minimize(cost_function,
                             np.random.rand(x_train.shape[0], 1),
                             jac=jac,
                             method='Newton-CG',
                             options={'gtol': epsilon})
    optimal_newton = newton_res.x
    newton_iterations = newton_res.nit
    return optimal_newton, newton_iterations


def bfgs_method():
    bfgs_res = sp.minimize(cost_function,
                           np.random.rand(x_train.shape[0], 1),
                           jac=jac,
                           method='BFGS',
                           options={'gtol': epsilon})
    optimal_bfgs = bfgs_res.x
    bfgs_iterations = bfgs_res.nit
    return optimal_bfgs, bfgs_iterations


def lbfgs_method():
    lbfgs_res = sp.minimize(cost_function,
                            np.random.rand(x_train.shape[0], 1),
                            jac=jac,
                            method='L-BFGS-B',
                            options={'gtol': epsilon})
    optimal_lbfgs = lbfgs_res.x
    lbfgs_iterations = lbfgs_res.nit
    return optimal_lbfgs, lbfgs_iterations


if __name__ == '__main__':
    main()
