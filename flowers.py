import numpy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

markers = ('s', '*', '^')
colors = ('blue', 'green', 'red')
cmap = ListedColormap(colors)
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
resolution = 0.01
