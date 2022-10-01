from src.metrics import *

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

def get_k_nearest_neighbors(point, data, k, metric):
    """
    Get the k nearest neighbors of a point in a dataset
    """
    distances = metric(point, data)
    return distances.argsort()[:k]

def get_k_nearest_neighbors_weights(point, data, k, metric, weights):
    """
    Get the k nearest neighbors of a point in a dataset weighing the neighbors
    """
    distances = metric(point, data)
    return distances.argsort()[:k], weights(distances)

def predict_inverse_LMAE(point, data, k, metric):
    neighbors = get_k_nearest_neighbors(point, data, k, metric)
    return np.average(data[neighbors], axis=0, weights=w_inverse_LMAE)

def error_bias(data, k, metric):
    error_bias = []
    for i in range(len(data)):
        neighbors = get_k_nearest_neighbors(data[i], data, k, metric)
        error_bias.append(np.sum(data[neighbors] - data[i])/k)
    return error_bias