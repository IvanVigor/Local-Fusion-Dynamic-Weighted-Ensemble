from .weights_functions import *
from sklearn.preprocessing import MinMaxScaler

class KWEnsembler():
    """
    KWEnsembler class

    This class implements the K-Weighted Ensembler model.
    It is an ensemble model that uses the k-nearest neighbors of a sample to predict its target value.
    The weights of the neighbors are calculated using a weight function.
    The bias of the neighbors can be added to the prediction.

    Parameters
    ----------

    :param k: number of neighbors to use
    :param bias: whether to add the bias of the neighbors to the prediction
    :param dist_metric: distance metric to use
    :type k: int
    :type bias: bool
    :type dist_metric: function

    :return: Predictions of the target values for the test set
    :rtype: bytearray
    """
    def __init__(self, k=5, bias='False', dist_metric=euclidean):

        self.k = k
        self.bias = bias
        self.dist_metric = dist_metric

    def fit(self, X_val, y_val):
        """
        Fits the ensemble by creating the search space

        Parameters
        ----------

        :param X_val: Validation set
        :param y_val: Validation set target values
        """
        self.X_val = X_val
        self.y_val = y_val

    def find_similar_neighbors(self, x, similar_space):
        """
        Finds the k nearest neighbors of x in the similar_space

        Parameters
        ----------

        :param x: Sample to find the neighbors of
        :param similar_space: Search space
        """

        sample = self.x_scaler.transform([x])
        distances = self.dist_metric(sample, similar_space)
        y_sorted = [y for _, y in sorted(zip(distances, distances.index))]
        return y_sorted[:self.k]

    def predict(self, X_test, features,  pred_columns,
                weight_function=w_inverse_LMAE, range_min=0, range_max=1):
        """
        Predicts the target values for the test set using the ensemble method

        Parameters
        ----------

        :param X_test: Test set
        :param features: Features of the test set
        :param pred_columns: Columns to predict
        :param weight_function: Weight function to use
        :param range_min: Minimum value of the target values
        :param range_max: Maximum value of the target values
        """

        self.x_scaler = MinMaxScaler([range_min, range_max])
        self.X_val[features] = self.x_scaler.fit_transform(self.X_val[features])
        predictions_ensembled = []

        for i in range(len(X_test)):

            weights = np.zeros(len(pred_columns))
            biases = np.zeros(len(pred_columns))
            neighbors = self.find_similar_neighbors(X_test[features].iloc[i], self.X_val[features])

            for idx, column in enumerate(pred_columns):
                preds_val = self.X_val.loc[neighbors][column]
                target_val = self.y_val.loc[neighbors]
                weights[idx] = weight_function(target_val, preds_val)
                if self.bias:
                    biases[idx]=sum((target_val.T - preds_val) / len(target_val))
            predictions_ensembled.append(sum(((X_test[pred_columns].iloc[i]-biases)*weights.T)) / sum(weights))

        return predictions_ensembled
