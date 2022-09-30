import pandas as pd
import numpy as np
import seaborn as sns
from metrics import *

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

# A function that predicts the value of a point given many different models provided as input
def predict(point, data, k, metric, weights):
    neighbors, weights = get_k_nearest_neighbors_weights(point, data, k, metric, weights)
    return np.average(data[neighbors], axis=0, weights=weights)

class KNeighborsSpotter():
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
        
    def fit(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        
    def predict(self, X_test, models_list):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_val)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_val, self.X_val))]
            neighbors.append(y_sorted[:self.k])
            
            for model in models_list:
                preds_val = model.predict(neighbors[2])
                target_val = neighbors[1]
                LMAE = mean_absolute_error(target_val, preds_val)
                e_P = target_val - preds_val
                
            
        return list(map(most_common, neighbors))