import numpy as np
import pandas as pd
import scipy.optimize as sp
from numpy import linalg
from typing import Tuple
import random
from functools import partial

from sklearn.model_selection import train_test_split

lr = 0.01
epsilon = 0.0001
data = pd.read_csv("resources/heart.csv")
y_data = data.target.values
x_data = data.drop(['target'], axis=1)
x_data_norm = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x_data_norm, y_data, test_size=0.2, random_state=0)
x_train = x_train.T
# x_train = np.c_[x_train, np.ones(x_train.shape[1])]
# np.append(x_train, np.ones((x_train.shape[1], 1)), axis=0)
x_train = np.vstack((x_train, np.ones(x_train.shape[1])))
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon

cost_function = lambda betta: 0.5 * np.mean(np.sum(((betta.T @ x_train) - y_train) ** 2))
jac = lambda betta: np.mean(np.sum(((x_train.T @ betta) - y_train) @ x_train.T))


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


def main():
    optimal_sr1, sr1_iterations = sr1_method()
    # y_prediction = predict(np.array([q[0] for q in optimal_sr1]))
    # print("Test Accuracy for SR1 method: {:.2f}% with {} iterations".format(
    #     (100 - np.mean(np.abs(y_prediction - y_test)) * 100), sr1_iterations))


if __name__ == '__main__':
    main()
