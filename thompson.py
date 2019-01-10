import numpy as np
import matplotlib.pyplot as plt

# setting the parameter

N = 10000
d = 9

# Creating the simulation

conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01]
X = np.array(np.zeros([N,d]))

for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i,j] = 1
            
