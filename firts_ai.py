

import numpy as np

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3

    (2 * .2) + (9 * .8) = 7.6
(2 * .6) + (9 * .3) = 3.9
(2 * .1) + (9 * .7) = 6.5

S(7.6) = 0.999499799
S(7.5) = 1.000553084
S(6.5) = 0.998498818

(.9994 * .4) + (1.000 * .5) + (.9984 * .9) = 1.79832

S(1.79832) = .8579443067



#weights
self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

def forward(self, X):
#forward propagation through our network
  self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
  self.z2 = self.sigmoid(self.z) # activation function
  self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
  o = self.sigmoid(self.z3) # final activation function
  return o

def sigmoid(self, s):
  # activation function
  return 1/(1+np.exp(-s))
print("Hello")
