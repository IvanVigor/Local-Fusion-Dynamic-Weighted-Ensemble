from ensemblem.model import KWEnsembler
from ensemblem.utils import *
from ensemblem.weights_functions import *

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

import numpy as np

"""
====================================================================
Example of using the KWEnsembler class
====================================================================

1. Load data
2. Split data into train, test and  neighbours-set
3. Train 3 different expert models on the train data
4. Generate predictions for the test data
5. Train the ensembler on the neighbours-set
6. Generate predictions for the test data coming from the ensembler
7. Compare the predictions from the ensembler with the predictions from the expert models
"""

# 1. Load data
california_housing = fetch_california_housing(as_frame=True)

# 2. Split data into train, validation test
X_train, y_train, X_neighbours, y_neighbours, X_test, y_test = split_sets(california_housing.frame.sample(frac=1), 0.70, 0.20, 0.10,
                                                                          california_housing.target_names[0])

# 3. Train 3 different expert models on train data
alphas = np.logspace(-3, 1, num=50)
model_one = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=alphas))

model_one.fit(X_train, y_train)
TreeRegressor_one = DecisionTreeRegressor(max_depth=3,
                                          random_state=123)
TreeRegressor_two = DecisionTreeRegressor(max_depth=5,
                                          random_state=123)

TreeRegressor_one.fit(X_train, y_train)
TreeRegressor_two.fit(X_train, y_train)


# 4. Generate predictions for the test and over neighbours-set
X_test["one_preds_rf"] = TreeRegressor_one.predict(X_test[california_housing.feature_names])
X_test["two_preds_rf"] = TreeRegressor_two.predict(X_test[california_housing.feature_names])
X_test["one_preds_r"] = model_one.predict(X_test[california_housing.feature_names])

X_neighbours["one_preds_rf"] = TreeRegressor_one.predict(X_neighbours[california_housing.feature_names])
X_neighbours["two_preds_rf"] = TreeRegressor_two.predict(X_neighbours[california_housing.feature_names])
X_neighbours["one_preds_r"] = model_one.predict(X_neighbours[california_housing.feature_names])

# 5. Train the ensembler on the neighbours-set
ensemble = KWEnsembler(50, bias=False)
ensemble.fit(X_neighbours, y_neighbours, features=california_housing.frame.drop(columns=[california_housing.target_names[0]]).columns)
results = ensemble.predict(X_test,
                california_housing.feature_names,
                ["one_preds_rf", "two_preds_rf",
                 "one_preds_r"],
                weight_function=w_inverse_log_LMAE)

# 6. Generate predictions for the test data coming from the ensembler
print(metrics_table(y_test, np.array(results), "Esemble"))
print(metrics_table(y_test, X_test["one_preds_rf"], "Tree"))
print(metrics_table(y_test, X_test["two_preds_rf"], "Tree"))
print(metrics_table(y_test, X_test["one_preds_r"], "RidgeCV"))

