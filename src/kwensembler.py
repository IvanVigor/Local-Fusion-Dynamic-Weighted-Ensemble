from src.weights_functions import *

class KWEnsembler():
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

    def predict(self, X_test, features,  pred_columns,
                weight_function=w_inverse_LMAE, bias=False):

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
            return (x[pred_columns] * np.array(weights).T+bias).sum(axis=1) / sum(weights)[0]
        else:
            return (x[pred_columns] * np.array(weights).T).sum(axis=1) / sum(weights)[0]
