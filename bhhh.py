import warnings
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy import optimize

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [6, 5]

DELTA = np.sqrt(np.finfo(float).eps)  # Config delta (for computing error)
EPS = 0.001
SUPPOSE_MIN = np.random.uniform(-0.5, 1.5, 2)
# source coefficients
a, b = np.random.uniform(0, 1, 2)
k = 100
# generate noisy data
x = np.arange(0, 1 + 1 / k, 1 / k)
d = np.random.normal(0, 1, k + 1)
y = a * x + b + d


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


def jacobian(params: Tuple[float, float], func):
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


def pain(func):
    """work"""
    a = []
    m = np.array([1, 1])  # SUPPOSE_MIN  np.array([1, 1])
    a.append(m)
    cnt = 0

    while cnt <= 5000:
        #         print(m)

        r = np.zeros((2, 2))

        e = jacobian(m, func).reshape(2, 1)
        for x1, y1 in zip(x, y):
            j = jacobian(m, simple(y1, x1)).reshape(2, 1)
            r += j.dot(j.T)

        m = m.reshape(2, 1)
        m = m - 0.01 * linalg.inv(r).dot(e)
        m = m.reshape(2)
        a.append(m)
        cnt += 1

    print(a[-1])
    print("number of iterations", cnt)
    return a


def main():
    # print(f'Start optimisation {SUPPOSE_MIN}')
    # print(f'a = {a}', f'b = {b}', sep='\n')

    # print(newton_method(linear))

    f = partial(middleware, func=linear)
    path = pain(f)
    print(path)
    plot_lines_lvl(linear, "qq", points_to_vectors(path), 3000, 100)


if __name__ == '__main__':
    main()
