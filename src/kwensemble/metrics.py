import pandas as pd
import numpy as np

def euclidean(point, data):
    return np.sqrt(np.sum((data - point)**2, axis=1))

def euclidean_v(x, y):
    """
    Euclidean distance is the square root of the sum of the squared differences of their coordinates
    """
    return np.sqrt(np.sum((x - y)**2))

def manhattan_v(x, y):
    """
    Manhattan distance is the sum of the absolute differences of their coordinates
    """
    return np.sum(np.abs(x - y))

def cosine_v(x, y):
    """
    Cosine distance is 1 - cosine similarity
    """
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

## REGRESSION METRICS ##
def mean_absolute_percentage_error(actual, predicted):
    """
    Local Mean Absolute Percentage Error (LMAPE)
    """
    return np.mean(np.abs((actual - predicted) / actual))

def mean_absolute_error(actual, predicted):
    """
    Mean Absolute Error (MAE)
    """
    return np.mean(np.abs(predicted-actual.T).T)

def root_mean_squared_error(actual, predicted):
    """
    Local Root Mean Squared Error (LRMSE)
    """
    return np.sqrt(np.mean((predicted-actual.T).T)**2)

def mean_squared_error(actual, predicted):
    """
    Mean Squared Error (MSE)
    """
    return np.mean((predicted-actual.T).T)**2

def root_mean_squared_log_error(actual, predicted):
    """
    Local Root Mean Squared Log Error (LRMSLE)
    """
    return np.sqrt(np.mean(np.square(np.log(predicted+1) - np.log(actual+1))))

# A function that create a table with results of the metrics
def metrics_table(actual, predicted):
    """
    Create a table with results of the metrics
    """
    metrics = pd.DataFrame()
    metrics['MAPE'] = mean_absolute_percentage_error(actual, predicted)
    metrics['MAE'] = mean_absolute_error(actual, predicted)
    metrics['RMSE'] = root_mean_squared_error(actual, predicted)
    metrics['MSE'] = mean_squared_error(actual, predicted)
    metrics['RMSLE'] = root_mean_squared_log_error(actual, predicted)
    return metrics