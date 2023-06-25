# Plot the sparsity pattern of a 2D array.
import matplotlib.pyplot as plt
import numpy as np


def plot_sparsity_pattern(array_2d, path, **args):
    """
    From https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.spy.html
    """
    x = array_2d
    plt.spy(x, **args)
    plt.savefig(path, dpi=300)
