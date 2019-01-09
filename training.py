import os
import numpy as np
import random as rn
import environment
import brain
import dqn

os.environ['PYTHONHASHEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETER
epsilon = .3
number_actions = 5
direction_boundary = (number_actions - 1) / 2
number_epochs = 1000
max_memory = 3000
batch_size = 512
temperature_step = 1.5
