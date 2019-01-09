#testing the AI
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

os.environ['PYTHONHASHEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING THE PARAMETER
number_actions = 5
direction_boundary = (number_actions - 1) / 2
temperature_step = 1.5

#BUILDING THE ENVOIRMENT BY CREATEING AN OBJECT
env = environment.Envronment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

#LOADING A PRE-TRAINED BRAIN

model = load_model("model.h5")

#CHOOSING THE MODE
train = False

#RUNNING A 1 YAER SIMULATION IN INPERENCE MODE
env.train = train
current_state, _, - = env.observe()
for timestep in range(0, 12 * 30 * 24 * 60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
    if (action - direction_boundary) < 0:
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    netx_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
    current_state = next_state

print('\n")
print("Epoch: {:03d}/{:03d}".format(epoch, number_epochs))
print("Total energy spend with an AI is {:.0f} ".format(env.total_energy_ai))
print("Total energy spend with no AI is {:.0f} ".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} ".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai))
