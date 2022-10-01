import pandas as pd
import numpy as np

def euclidean(point, data):
    return np.sqrt(np.sum((data - point)**2, axis=1))


def mean_absolute_error(actual, predicted):
    """
    Local Mean Absolute Error (LMAE)
    """
    return np.mean(np.abs(predicted-actual.T).T)


def root_mean_squared_error(actual, predicted):
    """
    Local Root Mean Squared Error (LRMSE)
    """
    return np.sqrt(np.mean((predicted-actual.T).T)**2)
