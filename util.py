import matplotlib.pyplot as plt
import numpy as np


def plot_cost_history(iterations, cost_history):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('Value of cost function')
    ax.set_xlabel('Iterations')
    _ = ax.plot(range(iterations), np.array(cost_history), 'b.')
    plt.show()
