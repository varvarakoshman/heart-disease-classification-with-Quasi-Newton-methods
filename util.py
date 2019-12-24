import matplotlib.pyplot as plt
import numpy as np


def plot_cost_history(iterations, cost_history):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('Value of cost function')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(iterations), np.array(cost_history), 'b.')
    plt.show()


def plot_all(regression, y, initial, bfgs, lbfgs, sr1, x=np.arange(0, 1 + 1 / 100, 1 / 100)):
    plt.scatter(x, y, s=60, color='blue', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, regression(initial[0], initial[1]), label='initial')
    plt.plot(x, regression(bfgs[0], bfgs[1]), label='BFGS')
    plt.plot(x, regression(lbfgs[0], lbfgs[1]), label='L-BFGS')
    plt.plot(x, regression(sr1[0], sr1[1]), label='SR1')
    # plt.plot(x, regression(bhhh[0], bhhh[1]), label='BHHH')
    plt.title('Linear regression', {'fontsize': 20}, pad=20)
    plt.legend(framealpha=1, frameon=True)
    plt.grid()
    plt.show()
