import pandas as pd
import numpy as np

def euclidean(point, data):
    """
    Euclidean distance is the square root of the sum of the squared differences of their coordinates

    :param point: point to calculate the distance
    :param data: data to calculate the distance

    :return: distance
    """
    return np.sqrt(np.sum((data - point)**2, axis=1))

def euclidean_v(x, y):
    """
    Vector Euclidean distance is the square root of the sum of the squared differences of their coordinates

    :param x: point to calculate the distance
    :param y: data to calculate the distance

    :return: distance
    """
    return np.sqrt(np.sum((x - y)**2))

def manhattan_v(x, y):
    """
    Vector Manhattan distance is the sum of the absolute differences of their coordinates

    :param x: point to calculate the distance
    :param y: data to calculate the distance
    """
    return np.sum(np.abs(x - y))

def cosine_v(x, y):
    """
    Vector Cosine distance is 1 - cosine similarity

    :param x: point to calculate the distance
    :param y: data to calculate the distance
    """
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def mean_absolute_percentage_error(actual, predicted):
    """
    Local Mean Absolute Percentage Error (LMAPE)

    :param actual: actual values
    :param predicted: predicted values
    """
    return np.mean(np.abs((actual - predicted) / actual))

def mean_absolute_error(actual, predicted):
    """
    Mean Absolute Error (MAE)

    :param actual: actual values
    :param predicted: predicted values

    """
    return np.mean(np.abs(predicted-actual.T).T)

def root_mean_squared_error(actual, predicted) -> float:
    """
    Local Root Mean Squared Error (LRMSE)
    """
    return np.sqrt(np.mean((predicted-actual.T).T)**2)

def mean_squared_error(actual, predicted) -> float:
    """
    Mean Squared Error (MSE)
    """
    return np.mean((predicted-actual.T).T)**2

def root_mean_squared_log_error(actual, predicted) -> float:
    """
    Local Root Mean Squared Log Error (LRMSLE)
    :param actual: actual values
    :param predicted: predicted values
    :return: RMSLE
    """
    return np.sqrt(np.mean(np.square(np.log(predicted+1) - np.log(actual+1))))


def metrics_table(actual, predicted, model_name) -> pd.DataFrame:
    """
    Create a table with pivot with results of multiple models and metrics

    :param actual: actual values
    :param predicted: predicted values
    :param model_name: name of the model

    :return: table with results
    """
    metrics = pd.DataFrame(columns=['Model', 'MAPE', 'MAE', 'RMSE', 'RMSLE'],
                           data=[[model_name, mean_absolute_percentage_error(actual, predicted),
                                  mean_absolute_error(actual, predicted),
                                  root_mean_squared_error(actual, predicted),
                                  root_mean_squared_log_error(actual, predicted)]])

    return metrics #.to_markdown()

