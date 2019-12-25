import warnings
from functools import partial
from typing import Tuple
import random
import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy import optimize

from util import plot_all

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [6, 5]

DELTA = np.sqrt(np.finfo(float).eps)  # Config delta (for computing error)
EPS = 0.001
epsilon = 0.0001
SUPPOSE_MIN = np.random.uniform(-0.5, 1.5, 2)
# source coefficients
a, b = np.random.uniform(0, 1, 2)
k = 100
# generate noisy data
x = np.arange(0, 1 + 1 / k, 1 / k)
d = np.random.normal(0, 1, k + 1)
y = a * x + b + d
stop_condition = lambda previous, current: np.linalg.norm(previous - current) < epsilon


def middleware(params, func):
    '''Format data as list'''
    return error(func, params[0], params[1])


def plot_lines_lvl(func, title, data, max_val=1000, step=50):
    '''Show lines of levels'''
    a_possible = np.arange(-0.5, 2, 0.01)
    b_possible = np.arange(-0.5, 2, 0.01)
    A, B = np.meshgrid(a_possible, b_possible)
    Z = np.zeros((len(b_possible), len(a_possible)))
    for i in range(len(b_possible)):
        for j in range(len(a_possible)):
            Z[i][j] = error(func, a_possible[j], b_possible[i])

    levels = [i for i in range(0, max_val, step)]
    plt.contour(A, B, Z, levels, colors='k')
    contour_filled = plt.contourf(A, B, Z, levels, cmap="RdBu_r")
    plt.colorbar(contour_filled)

    plt.title(title, {'fontsize': 20}, pad=25)
    plt.xlabel('a', {'fontsize': 15})
    plt.ylabel('b', {'fontsize': 15})

    plt.plot(data[0], data[1], color='red', label='way to minimum')
    plt.scatter(data[0], data[1], s=30, color='red', edgecolors='black')

    plt.legend(loc='best')
    plt.show()


def jacobian(params, func):
    return optimize.approx_fprime(params, func, (DELTA, DELTA))


def hessian(params: Tuple[float, float], func):
    current_jac = partial(jacobian, func=func)

    diff_func_a = lambda point: current_jac(point)[0]
    diff_func_b = lambda point: current_jac(point)[1]

    return (optimize.approx_fprime(params, diff_func_a, (100 * DELTA, 100 * DELTA)),
            optimize.approx_fprime(params, diff_func_b, (100 * DELTA, 100 * DELTA)))


def points_to_vectors(points):
    '''Convert from list of points to pair of lists coordinates'''
    return [[i for i, _ in points], [j for _, j in points]]


def error(func, a, b):
    return np.sum((func(a, b) - y) ** 2)


def linear(a, b):
    return a * x + b


def newton_method(func):
    error_func = partial(middleware, func=func)
    current_jac = partial(jacobian, func=error_func)
    current_hess = partial(hessian, func=error_func)

    result = optimize.minimize(error_func, method='Newton-CG', jac=current_jac, hess=current_hess, x0=SUPPOSE_MIN,
                               options={'xtol': EPS, 'return_all': True})

    return result.x, result.nfev, result.njev, result.nit, result.allvecs, result.nhev


def simple(y, x):
    return lambda p: ((p[0] * x + p[1]) - y) ** 2


def bhhh(func):
    path = []
    prev = np.array([np.inf, np.inf])
    m = np.array([2, 2])  # SUPPOSE_MIN  np.array([1, 1])
    path.append(m)
    cnt = 0
    while np.linalg.norm(prev - m) > epsilon and cnt <= 5000:
        prev = m
        r = np.zeros((2, 2))
        e = jacobian(m, func).reshape(2, 1)
        for x1, y1 in zip(x, y):
            j = jacobian(m, simple(y1, x1)).reshape(2, 1)
            r += j.dot(j.T)
        optimal = lambda alpha: func(m - alpha * linalg.inv(r).dot(e).reshape(2))
        alpha, cnt_call = golden_section(optimal)
        m = m - alpha * linalg.inv(r).dot(e).reshape(2)
        path.append(m)
        cnt += 1
    return path, cnt, m


def golden_section(func, a=0, b=3):
    gold = (3 - math.sqrt(5)) / 2

    x1 = gold * (b - a) + a
    x2 = gold * (a - b) + b

    f1 = func(x1)
    f2 = func(x2)
    iters = 0
    while abs(b - a) >= EPS:
        iters += 1  # Number of calls is two more than number of iterations
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = gold * (b - a) + a
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = gold * (a - b) + b
            f2 = func(x2)

    return (b + a) / 2, iters


def sr1_method():
    path = []
    x_k = np.array([random.random(), random.random()])  # starting point
    hessian_inverse = np.eye(len(x_k), dtype=int)  # initial value of hessian is set to Identity matrix
    iterations = 0
    alpha_prev = 0.001
    cost_function = partial(middleware, func=linear)
    while True:
        gfk = jacobian(x_k, cost_function)
        p_k = - np.dot(hessian_inverse, gfk)
        line_search = optimize.optimize.line_search(cost_function, partial(jacobian, func=cost_function), x_k, p_k)
        alpha_optimal = line_search[0]
        if alpha_optimal is None:
            alpha_optimal = alpha_prev
        x_kp1 = x_k + alpha_optimal * p_k
        delta_x = x_kp1 - x_k
        y_k = jacobian(x_kp1, cost_function) - gfk
        w = delta_x - np.dot(hessian_inverse, y_k)
        if np.absolute(np.dot(w.T, y_k)) >= 0.001 * np.linalg.norm(w.T) * np.linalg.norm(y_k):
            r0 = 1.0 / (np.dot(w.T, y_k))
            hessian_inverse = hessian_inverse + r0 * np.dot(w, w.T)
        iterations += 1
        alpha_prev = alpha_optimal
        path.append(x_kp1)
        if stop_condition(x_k, x_kp1):
            break
        else:
            x_k = x_kp1
    return x_kp1, iterations, path


def main():
    print(f'a = {a}', f'b = {b}', sep='\n')

    # Symmetric rank 1 results
    optimal_sr1, sr1_iterations, path = sr1_method()
    print("root with SR1 method: ", optimal_sr1)
    print("# of iterations with SR1 method: ", sr1_iterations)
    plot_lines_lvl(linear, "SR1", points_to_vectors(path), 3000, 100)

    # Newton results
    x, nfev, njev, nit, allvecs, nhev = newton_method(linear)
    print("root with Newton method: ", x)
    print("# of iterations with Newton method: ", nit)
    plot_lines_lvl(linear, "Newton", points_to_vectors(allvecs), 3000, 100)

    error_func = partial(middleware, func=linear)
    current_jac = partial(jacobian, func=error_func)

    # BFGS results
    bfgs_res = optimize.minimize(error_func,
                                 np.array([random.random(), random.random()]),
                                 jac=current_jac,
                                 method='BFGS',
                                 options={'gtol': epsilon, 'return_all': True})
    print("root with BFGS method: ", bfgs_res.x)
    print("# of iterations with BFGS method: ", bfgs_res.nit)
    plot_lines_lvl(linear, "BFGS", points_to_vectors(bfgs_res.allvecs), 3000, 100)

    # L-BFGS results
    lbfgs_res = optimize.minimize(error_func,
                                  np.array([random.random(), random.random()]),
                                  jac=current_jac,
                                  method='L-BFGS-B',
                                  options={'gtol': epsilon})
    print("root with L-BFGS method: ", lbfgs_res.x)
    print("# of iterations with L-BFGS method: ", lbfgs_res.nit)

    # bhhh algorithm results
    f = partial(middleware, func=linear)
    path, cnt, m = bhhh(f)
    plot_lines_lvl(linear, "bhhh", points_to_vectors(path), 3000, 100)
    print("root with BHHH method: ", m)
    print("# of iterations with BHHH method: ", cnt)

    plot_all(linear, y, [a, b], bfgs_res.x, lbfgs_res.x, optimal_sr1, m)


if __name__ == '__main__':
    main()
