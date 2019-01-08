import numpy as np

class DQN(object):
    
    #INTRODUCING AND INITIALIZZIN ALL THE PARAMETERS AND VARIABLES OF THE DQN
    def __init__(self, max_memory = 100, discount = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount
    #MAKING A METOD THAT BUILDS THE MEMORY IN EXPERIENCE REPLAY
    
    # MAKING A METHOD THAT BUILDS TWO BATHCES OF 10 INOUTS AND 10 TARGERS BY EXTRANCTING 10 TRANSITIONS
