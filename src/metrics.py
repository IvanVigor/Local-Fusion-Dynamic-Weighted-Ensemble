import pandas as pd
import numpy as np

def euclidean(point, data):
    return np.sqrt(np.sum((data - point)**2, axis=1))

def w_inverse_LMAE(actual, predicted):
    """
    Inverse Local MAE
    """
    return 1/mean_absolute_error(actual, predicted)

def w_inverse_log_LMAE(residuals):
    """
    Inverse Log Local MAE
    """
    np.log(max(abs(residuals))/mean_absolute_error(residuals))


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
