import pandas as pd
import numpy as np
from src.metrics import *

# A function which return the most similar elements to a point
def get_k_nearest_neighbors(point, data, k, metric):
    distances = metric(point, data)
    return distances.argsort()[:k]

def get_k_nearest_neighbors_weights(point, data, k, metric, weights):
    distances = metric(point, data)
    return distances.argsort()[:k], weights(distances)

# A function that predicts the value of a point given many different models provided as input using the inverse of the LMAE as weights
def predict_inverse_LMAE(point, data, k, metric):
    neighbors = get_k_nearest_neighbors(point, data, k, metric)
    return np.average(data[neighbors], axis=0, weights=w_inverse_LMAE)

# A function that evalaute the error bias associated to each machine learning forecast diveded by the number of forecasts
def error_bias(data, k, metric):
    error_bias = []
    for i in range(len(data)):
        neighbors = get_k_nearest_neighbors(data[i], data, k, metric)
        error_bias.append(np.sum(data[neighbors] - data[i])/k)
    return error_bias


class KNeighborsSpotter():
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val

    def find_similar_neighbors(self, x):

        distances = self.dist_metric(x, self.X_val)
        y_sorted = [y for _, y in sorted(zip(distances, distances.index))]
        return y_sorted[:self.k]

    def predict(self, X_test, features,  pred_columns, target_column, weight_function=w_inverse_LMAE, bias=False):

        weights = []
        biases = []

        for x in X_test.iterrows():
            x = pd.DataFrame(x[1]).T
            neighbors = self.find_similar_neighbors(x[features].fillna(value=0))

        for column in pred_columns:
            preds_val = self.X_val.loc[neighbors][column]
            target_val = self.y_val.loc[neighbors]
            w = weight_function(target_val, preds_val)
            weights.append(w)
            biases = (target_val - preds_val) / len(target_val)
            if bias:
                return (x[pred_columns] * np.array(weights).T+bias).sum(axis=1) / sum(weights)
            else:
                return (x[pred_columns] * np.array(weights).T).sum(axis=1) / sum(weights)
