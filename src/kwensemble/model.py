from src.kwensemble.weights_functions import *

class KWEnsembler():
    def __init__(self, k=5, bias='False', dist_metric=euclidean):
        self.k = k
        self.bias = bias
        self.dist_metric = dist_metric

    def fit(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val

    def find_similar_neighbors(self, x):

        distances = self.dist_metric(x, self.X_val)
        y_sorted = [y for _, y in sorted(zip(distances, distances.index))]
        return y_sorted[:self.k]

    def predict(self, X_test, features,  pred_columns,
                weight_function=w_inverse_LMAE, bias=False):

        predictions_ensembled = []

        for i in range(len(X_test)):

            weights = np.zeros(len(pred_columns))
            biases = np.zeros(len(pred_columns))

            neighbors = self.find_similar_neighbors(X_test[features].iloc[i])

            for _, column in enumerate(pred_columns):
                preds_val = self.X_val.loc[neighbors][column]
                target_val = self.y_val.loc[neighbors]
                weights[_] = weight_function(target_val, preds_val)
                if self.bias:
                    biases[_]=((target_val.T - preds_val) / len(target_val)).sum(axis=1)
            predictions_ensembled.append(sum((X_test[pred_columns].iloc[i] * np.array(weights).T+biases)) / sum(weights))

        return predictions_ensembled
