import numpy as np
import pandas as pd
import scipy.optimize as sp
from numpy import linalg
from typing import Tuple

from sklearn.model_selection import train_test_split

from util import plot_cost_history

# dataset and learning rate are available globally
lr = 0.01
epsilon = 0.0001
data = pd.read_csv("resources/heart.csv")
y_data = data.target.values
x_data = data.drop(['target'], axis=1)
x_data_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x_data_norm, y_data, test_size=0.2, random_state=0)
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon
# cost_function = lambda point: np.mean(-(y_train * np.log(sigmoid(np.dot(point.T, x_train))) + (1 - y_train) * np.log(
#     1 - sigmoid(np.dot(point.T, x_train)))))
jac = lambda point: np.dot(x_train, (sigmoid(np.dot(point.T, x_train)) - y_train).T) / x_train.shape[1]

functional_linear = lambda betta: 0.5 * np.sum(np.array([np.square(y_train - (np.dot(x_train.T, betta) + ))]))



def main():
    # parameters_gd, iterations_gd = gradient_descent()
    # y_prediction = predict(np.array([q[0] for q in parameters_gd]))
    # print("Test Accuracy for GD: {:.2f}% with {} iterations".format(
    #     (100 - np.mean(np.abs(y_prediction - y_test)) * 100), iterations_gd))

    # optimal_newton, newton_iterations = newton_method()
    # y_prediction = predict(optimal_newton)
    # print("Test Accuracy for Newton method: {:.2f}% with {} iterations".format(
    #     (100 - np.mean(np.abs(y_prediction - y_test)) * 100), newton_iterations))

    # optimal_bfgs, bfgs_iterations = bfgs_method()
    # y_prediction = predict(optimal_bfgs)
    # print("Test Accuracy for BFGS method: {:.2f}% with {} iterations".format(
    #     (100 - np.mean(np.abs(y_prediction - y_test)) * 100), bfgs_iterations))

    # optimal_lbfgs, lbfgs_iterations = lbfgs_method()
    # y_prediction = predict(optimal_lbfgs)
    # print("Test Accuracy for Limited memory BFGS method: {:.2f}% with {} iterations".format(
    #     (100 - np.mean(np.abs(y_prediction - y_test)) * 100), lbfgs_iterations))

    optimal_sr1, sr1_iterations = sr1_method()
    y_prediction = predict(np.array([q[0] for q in optimal_sr1]))
    print("Test Accuracy for SR1 method: {:.2f}% with {} iterations".format(
        (100 - np.mean(np.abs(y_prediction - y_test)) * 100), sr1_iterations))

    # optimal_bhhh, bhhh_iterations = bhhh_method()
    # y_prediction = predict(optimal_bhhh)
    # print("Test Accuracy for bhhh method: {:.2f}% with {} iterations".format(
    #     (100 - np.mean(np.abs(y_prediction - y_test)) * 100), bhhh_iterations))


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
        cost = cost_function(prev)
        gradient = np.dot(x_train, (sigmoid(np.dot(prev.T, x_train)) - y_train).T) / x_train.shape[1]
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


def sr1_method():
    x_k = np.array([j[0] for j in np.random.rand(x_train.shape[0], 1)])  # starting point
    # x_k = np.random.rand(x_train.shape[0], 1)
    hessian_inverse = np.eye(x_train.shape[0], dtype=int)  # initial value of hessian is set to Identity matrix
    iterations = 0
    # alpha_optimal = 0.001
    while True:
        gfk = jac(x_k)
        p_k = - np.dot(hessian_inverse, gfk)
        line_search = sp.optimize.line_search(cost_function, jac, x_k, p_k)
        alpha_optimal = line_search[0]
        x_kp1 = x_k + alpha_optimal * p_k
        delta_x = x_kp1 - x_k
        y_k = jac(x_kp1) - gfk
        w = delta_x - np.dot(hessian_inverse, y_k)
        if np.absolute(np.dot(w.T, y_k)) >= 0.001 * np.linalg.norm(w.T) * np.linalg.norm(y_k):
            r0 = 1.0 / (np.dot(w.T, y_k))
            hessian_inverse = hessian_inverse + r0 * np.dot(w, w.T)
        iterations += 1
        if stop_condition(x_k, x_kp1):
            break
        else:
            # print(np.linalg.norm(x_k - x_kp1))
            x_k = x_kp1
    return x_kp1, iterations


# TODO: поменять цикл с 5000 итерациями на цикл с точностью 10е-3
# def simple(y, x):
#     return lambda p: ((p[0] * x + p[1]) - y) ** 2
#
#
# def bhhh_method():
#     prev = np.random.rand(x_train.shape[0], 1)
#     a = []
#     # m = np.array([1, 1])  # SUPPOSE_MIN  np.array([1, 1])
#     a.append(prev)
#     iterations = 0
#     while iterations <= 5000:
#         r = np.zeros((2, 2))
#         e = jac(prev)
#         for x1, y1 in zip(x, y): # что с x, y
#             j = jacobian(prev, simple(y1, x1)) # что за simple
#             r += j.dot(j.T)
#         prev = prev.reshape(2, 1) # что с размерностями
#         prev = prev - 0.01 * linalg.inv(r).dot(e)
#         prev = prev.reshape(2)
#         a.append(prev)
#         iterations += 1
#     print(a[-1])
#     return a


if __name__ == '__main__':
    main()
