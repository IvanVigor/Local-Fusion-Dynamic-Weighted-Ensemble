from src.kwensemble.model import KWEnsembler
from src.kwensemble.utils import *

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
2. Split data into train and test
3. Train 3 different expert models on the train data
4. Generate predictions for the test data
5. Train the ensembler on the train data
6. Generate predictions for the test data coming from the ensembler
7. Compare the predictions from the ensembler with the predictions from the expert models
"""

# 1. Load data
california_housing = fetch_california_housing(as_frame=True)

# 2. Split data into train, validation test
X_train, y_train,X_validation, y_validation, X_test, y_test = split_sets(california_housing.frame, 0.80, 0.10, 0.10, california_housing.feature_names, california_housing.target_names)

# 3. Train 3 different expert models on the train data
alphas = np.logspace(-3, 1, num=30)
model = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=alphas))
TreeRegressor_one = DecisionTreeRegressor(max_depth=3,
                                          random_state=123)
TreeRegressor_two = DecisionTreeRegressor(max_depth=5,
                                          random_state=123)

model = make_pipeline(StandardScaler(),
                      RidgeCV(alphas=alphas))

TreeRegressor_one.fit(X_train, y_train)
TreeRegressor_two.fit(X_train, y_train)

# 4. Generate predictions for the test data
X_test["one_preds"] = TreeRegressor_one.predict(X_test[california_housing.feature_names])
X_test["two_preds"] = TreeRegressor_two.predict(X_test[california_housing.feature_names])
X_test["three_preds"] = TreeRegressor_two.predict(X_test[california_housing.feature_names])

X_validation["one_preds"] = TreeRegressor_one.predict(X_validation[california_housing.feature_names])
X_validation["two_preds"] = TreeRegressor_two.predict(X_validation[california_housing.feature_names])
X_validation["three_preds"] = TreeRegressor_two.predict(X_validation[california_housing.feature_names])

# 5. Train the ensembler on the train data
ensemble = KWEnsembler(5)
ensemble.fit(X_validation, y_validation)
results = ensemble.predict(X_test.head(100),
                california_housing.feature_names,
                ["one_preds", "two_preds"]
                )

print(np.mean(results-y_test.head(100).values))
print(np.mean(np.mean(X_test["one_preds"])-y_test.head(100).values))
print(np.mean(np.mean(X_test["two_preds"]) - y_test.head(100).values))