import numpy as np

class DQN(object):
    
    #INTRODUCING AND INITIALIZZIN ALL THE PARAMETERS AND VARIABLES OF THE DQN
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
    #MAKING A METOD THAT BUILDS THE MEMORY IN EXPERIENCE REPLAY
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]
    # MAKING A METHOD THAT BUILDS TWO BATHCES OF 10 INOUTS AND 10 TARGERS BY EXTRANCTING 10 TRANSITIONS
    def get_batch(self, model, batch_size =10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros(min(len_memory, batch_size), num_inputs)
        targets = np.zeros(min(len_memory, batch_size), num_inputs)
        for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
    
    
    
