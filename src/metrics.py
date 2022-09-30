import pandas as pd
import numpy as np
import seaborn as sns

def euclidean(point, data):
    return np.sqrt(np.sum((data - point)**2, axis=1))

def w_inverse_LMAE(lmae):
    """
    Inverse Local MAE
    """
    return 1/lmae

def w_inverse_log_LMAE(residuals):
    """
    Inverse Log Local MAE
    """
    np.log(max(abs(residuals))/mean_absolute_error(residuals))


def mean_absolute_error(actual, predicted):
    """
    Local Mean Absolute Error (LMAE)
    """
    return np.mean(np.abs(predicted - actual))


def root_mean_squared_error(actual, predicted):
    """
    Local Root Mean Squared Error (LRMSE)
    """
    return np.sqrt(np.mean((predicted - actual)**2))


def weighted_mean_absolute_error(actual, predicted, weights):
    """
    Weighted Local Mean Absolute Error (WLMAE)
    """
    return np.average(np.abs(predicted - actual), weights=weights)