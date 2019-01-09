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

#BUILDING THE ENVOIRMENT BY CREATEING AN OBJECT
env = environment.Envronment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#BULDING THE BRAIN 
brain = brain.Brain(learning_rate = 0.00001, number_actions = 5)
# BUILDING THE DQN MODEL

dqn = dqn.DQN(max_memory = max_memory, discount = 0.9)

#CHOOSING THE MODE
train = True

#TRAINING THE AI

env.train = train
model = brain.model
if env.train:
    for epoch in range(1, number_epochs):
        totwal_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        while (not game_over) and timestep <= 5 * 30 * 24 * 60:
            # PLAYNING THE NEXT ACTION BY EXPLERATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if action - direction_boundary < 0:
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temperature_step
            #PLAYING THE NEXT ACTION BY INFERENCE
            
            #UPDATING THE ENVIROMENT AND REACHING THE NEXT STATE
            
            #STORING THIS NEW TRANSITION INTO MEMORY
            
            #GATHERING IN TWO SEPERATE BATCHES THE INPUST AND THE TARGETS
            
            #COMPUTING THE LOSS OVER TWO WHOLE BATCHES OF INOUST AND TARGETS
            
