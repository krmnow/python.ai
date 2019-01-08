from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

#BUILDING THE BRAIN

class Brain(object):
    
    def __init__(self, learning_rate = 0.001, number_actions = 5)
