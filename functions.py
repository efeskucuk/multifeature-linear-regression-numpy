import numpy as np
import math
import pandas as pd

# Set up data set
def setupDataSet(trainingDataRatio):
    # Read CSV and move to numpy array
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
            'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B1000', 'LSTAT', 'MEDV']
    df = pd.read_csv('housing.csv', header=None, names=columns, delimiter=r"\s+")
    numpyArray = np.array(df.values)
    
    datasetLength = len(numpyArray)

    # Normalize features
    normalizedFeatures = getNormalizedSet(numpyArray[0:, 0:13])

    # t for training set, v for validation set
    tSetLength = math.floor((datasetLength / 100) * trainingDataRatio)
    tPrices = numpyArray[:tSetLength, 13:14]
    tFeatures = normalizedFeatures[:tSetLength, 0:13]
    tFeatures = np.c_[ tFeatures, np.ones(tSetLength)] # for b

    vPrices = numpyArray[tSetLength:, 13:14]
    vFeatures = normalizedFeatures[tSetLength:, 0:13]
    vSetLength = len(vPrices)
    vFeatures = np.c_[ vFeatures, np.ones(vSetLength)] # for b

    return tSetLength, tPrices, tFeatures, vSetLength, vPrices, vFeatures


# Output of this function is a m x 1 matrix, every row is one prediction
def f_wb(weights, features):
    return np.dot(features, weights)

# Calculates how far off we are from the minimum
def cost_calc(w, x, y, m):
    subtraction = np.subtract(f_wb(w, x), y)
    squared = np.square(subtraction)
    summation = np.sum(squared)
    return summation / (m * 2)

# Calculates improved weights
def gradient_descent(w, x, y, m, a):
    subtraction = np.subtract(f_wb(w, x), y)
    multiplication = subtraction * x
    summation = np.sum(multiplication, axis=0, keepdims=True)

    # Every column of the variable below is derivative for one weight
    division = np.divide(summation, m)

    # Multiply this by the learning rate and subtract it from weights
    timesLearningRate = np.multiply(division, a)
    transposed = np.transpose(timesLearningRate)
    new_weights = np.subtract(w, transposed)

    return new_weights

# Trains the model against the set
def train(trainingFeatures, trainingPrices, trainingSetLength, weights, a, iterations):
    trainedWeights = weights
    for i in range(iterations):
        trainedWeights = gradient_descent(
            trainedWeights, trainingFeatures, trainingPrices, trainingSetLength, a)

    return trainedWeights

def getNormalizedSet(tFeatures):
    average = np.average(tFeatures, axis=0)
    maxMinusMin = np.subtract(np.max(tFeatures, axis=0), np.min(tFeatures, axis=0))
    return np.divide(np.subtract(tFeatures, average), maxMinusMin)

def print_cost(tSetLength, tPrices, tFeatures, vSetLength, vPrices, vFeatures, weights):
    print('=>\nValidation data set error: {}\nTraining data set error: {}\n'.format(cost_calc(
    weights, vFeatures, vPrices, vSetLength),
    cost_calc(
    weights, tFeatures, tPrices, tSetLength)))
