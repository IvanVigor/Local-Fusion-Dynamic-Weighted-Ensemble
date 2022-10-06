from src.ensemblem.weights_functions import *
from sklearn.preprocessing import MinMaxScaler

class KWEnsembler():
    def __init__(self, k=5, bias='False', dist_metric=euclidean):
        self.k = k
        self.bias = bias
        self.dist_metric = dist_metric

    def fit(self, X_val, y_val, range_min=0, range_max=1):
        """
        Fits the ensemble to the validation set
        """

        self.X_val = X_val
        self.y_val = y_val

    def find_similar_neighbors(self, x, similar_space):
        """
        Finds the k nearest neighbors of x in the similar_space
        """

        sample = self.x_scaler.transform([x])
        distances = self.dist_metric(sample, similar_space)
        y_sorted = [y for _, y in sorted(zip(distances, distances.index))]
        return y_sorted[:self.k]

    def predict(self, X_test, features,  pred_columns,
                weight_function=w_inverse_LMAE, range_min=0, range_max=1):
        """
        Predicts the target values for the test set using the ensemble
        """
        self.x_scaler = MinMaxScaler([range_min, range_max])
        self.X_val[features] = self.x_scaler.fit_transform(self.X_val[features])
        predictions_ensembled = []

        for i in range(len(X_test)):

            weights = np.zeros(len(pred_columns))
            biases = np.zeros(len(pred_columns))

            neighbors = self.find_similar_neighbors(X_test[features].iloc[i], self.X_val[features])

            for _, column in enumerate(pred_columns):
                preds_val = self.X_val.loc[neighbors][column]
                target_val = self.y_val.loc[neighbors]
                weights[_] = weight_function(target_val, preds_val)
                if self.bias:
                    biases[_]=sum((target_val.T - preds_val) / len(target_val))
            predictions_ensembled.append(sum(((X_test[pred_columns].iloc[i]-biases)*weights.T)) / sum(weights))

        return predictions_ensembled
