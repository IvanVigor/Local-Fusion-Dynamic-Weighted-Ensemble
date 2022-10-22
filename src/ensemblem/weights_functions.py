from .metrics import *

def w_inverse_LMAE(actual, predicted):
    """
    Inverse Local MAE

    :param actual: actual values
    :param predicted: predicted values
    """
    return 1/mean_absolute_error(actual, predicted)

def w_inverse_log_LMAE(actual, predicted):
    """
    Inverse Log Local MAE

    :param actual: actual values
    :param predicted: predicted values

    :return: inverse log local MAE
    """
    return np.log(max(abs(actual-predicted))/mean_absolute_error(actual, predicted))

def get_k_nearest_neighbors(point, data, k, metric):
    """
    Get the k nearest neighbors of a point in a dataset

    :param point: point to calculate the distance
    :param data: data to calculate the distance
    :param k: number of neighbors
    :param metric: distance metric
    """
    distances = metric(point, data)
    return distances.argsort()[:k]

def get_k_nearest_neighbors_weights(point, data, k, metric, weights):
    """
    Get the k nearest neighbors of a point in a dataset weighing the neighbors.
    Parameters: point, data, k, metric, weights

    :param point: point to calculate the distance
    :param data: data to calculate the distance
    :param k: number of neighbors
    :param metric: distance metric
    :param weights: weights of the neighbors

    :return: k nearest neighbors
    """

    distances = metric(point, data)
    return distances.argsort()[:k], weights(distances)

def predict_inverse_LMAE(point, data, k, metric):
    """
    Predict the target value of a point using the inverse LMAE
    """
    neighbors = get_k_nearest_neighbors(point, data, k, metric)
    return np.average(data[neighbors], axis=0, weights=w_inverse_LMAE)

def error_bias(data, k, metric):
    """
    Calculate the bias of the error

    :param data: data to calculate the distance
    :param k: number of neighbors
    :param metric: distance metric

    :return: bias of the error
    """
    error_bias = []
    for i in range(len(data)):
        neighbors = get_k_nearest_neighbors(data[i], data, k, metric)
        error_bias.append(np.sum(data[neighbors] - data[i])/k)
    return error_bias