import numpy as np
from functions import *

# Weights and constant at the end, model : w_1*x_1 + w_2*x_2 + ... + b
weights = np.zeros((14, 1))

# Learning rate
a = 0.001

# Training data set and Validation data set setup, 80/20
tSetLength, tPrices, tFeatures, vSetLength, vPrices, vFeatures = setupDataSet(trainingDataRatio=80)

# Print cost
print_cost(tSetLength, tPrices, tFeatures, vSetLength, vPrices, vFeatures, weights)

# Train
weights = train(tFeatures, tPrices, tSetLength, weights, a, iterations=100000)

# Print cost again
print("=>\nModel trained\n")
print_cost(tSetLength, tPrices, tFeatures, vSetLength, vPrices, vFeatures, weights)
