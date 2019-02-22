import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def tahh(x):
    return (1.0 - numpy.exp(-2*x))/(1.0 - numpy.exp(-2*x))
def tahn_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))
