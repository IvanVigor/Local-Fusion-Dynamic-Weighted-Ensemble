import pandas
from .weights_functions import *
from sklearn.preprocessing import MinMaxScaler
from typing import List

class KWEnsembler():
    """
    KWEnsembler class

    This class implements the K-Weighted Ensembler model.
    It is an ensemble model that uses the k-nearest neighbors of a sample to predict its target value.
    The weights of the neighbors are calculated using a weight function.
    The bias of the neighbors can be added to the prediction.

    :param k: number of neighbors to use
    :param bias: whether to add the bias of the neighbors to the prediction
    :param dist_metric: distance metric to use

    :return: Predictions of the target values for the test set
    :rtype: bytearray
    """
    def __init__(self, k: int=5, bias:bool='False', dist_metric=euclidean):

        self.k = k
        self.bias = bias
        self.dist_metric = dist_metric

    def fit(self, X_val: pandas.DataFrame,
                    y_val: pandas.DataFrame):
        """
        Fits the ensemble by creating the search space

        Parameters
        ----------

        :param X_val: Validation set
        :param y_val: Validation set target values
        """
        self.X_val = X_val
        self.y_val = y_val

    def _find_similar_neighbors(self, test_sample: pandas.Series,
                                     similar_space: pandas.DataFrame) -> List:
        """
        Finds the k nearest neighbors of x in the similar_space

        Parameters
        ----------

        :param x: Sample to find the neighbors of
        :param similar_space: Search space

        :return: Indices of the k nearest neighbors
        """

        element = self.x_scaler.transform([test_sample])
        distances = self.dist_metric(element, similar_space)
        y_sorted = [y for _, y in sorted(zip(distances, distances.index))]
        return y_sorted[:self.k]

    def predict(self, X_test: pandas.DataFrame,
                        features: List,
                        pred_columns: List,
                        weight_function=w_inverse_LMAE,
                        range_min: int=0, range_max: int=1) -> List:
        """
        Predicts the target values for the test set using the ensemble method

        :param X_test: Test set
        :param features: Features of the test set
        :param pred_columns: Columns to predict
        :param weight_function: Weight function to use
        :param range_min: Minimum value of minmax scaling
        :param range_max: Maximum value of minmax scaling

        :return: Predictions of the target values for the test set
        """

        self.x_scaler = MinMaxScaler([range_min, range_max])
        self.X_val[features] = self.x_scaler.fit_transform(self.X_val[features])
        predictions_ensembled = []

        for i in range(len(X_test)):

            _weights = np.zeros(len(pred_columns))
            _biases = np.zeros(len(pred_columns))
            _neighbors = self._find_similar_neighbors(X_test[features].iloc[i],
                                                     self.X_val[features])

            for idx, column in enumerate(pred_columns):
                preds_val = self.X_val.loc[_neighbors][column]
                target_val = self.y_val.loc[_neighbors]
                _weights[idx] = weight_function(target_val, preds_val)
                if self.bias:
                    _biases[idx]=sum((target_val.T - preds_val) / len(target_val))
            predictions_ensembled.append(sum(((X_test[pred_columns].iloc[i]-_biases)*_weights.T)) / sum(_weights))

        return predictions_ensembled
