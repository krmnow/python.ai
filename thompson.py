import numpy as np
import matplotlib.pyplot as plt
import random

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
            
# implementing a random strategy and Thompson sampling
strategies_selected_rs = []
strategies_selected_ts = []
total_reward_rs = 0
total_reward_ts = 0
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d

for n in range(0, N):
    # Random Strategy
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs
    #Thompson Sampling
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts =1
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1
    else:
        numbers_of_rewards_0[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts
    
# cumputing the absolute and relative return
    
absolute_return = (total_reward_ts - total_reward_rs)*100
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs
print("Absolute return: {:.0f} $".format(absolute_return))
print("Relative return: {:.0f} %".format(relative_return))

plt.hist(strategies_selected_ts)
plt.title("Histogram of selection")
plt.xlabel("Strategy")
plt.ylabel("Number of times the strategy was selected")
plt.show()
