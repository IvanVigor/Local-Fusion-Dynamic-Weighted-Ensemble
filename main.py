from src.kwensembler import KWEnsembler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
import numpy as np

if __name__ == "__main__":

    california_housing = fetch_california_housing(as_frame=True)

    training = california_housing.frame[california_housing.feature_names]
    target = california_housing.frame[california_housing.target_names]

    X_train, X_test, y_train, y_test = train_test_split(training, target, random_state=1234, test_size=0.20)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, random_state=1234, test_size=0.20)

    alphas = np.logspace(-3, 1, num=30)
    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=alphas))

    tree_one = DecisionTreeRegressor(max_depth=3, random_state=0)
    tree_two = DecisionTreeRegressor(max_depth=10, random_state=0)

    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=alphas))

    tree_one.fit(X_train, y_train)
    tree_two.fit(X_train, y_train)

    X_test["one_preds"] = tree_one.predict(X_test[california_housing.feature_names])
    X_test["two_preds"] = tree_two.predict(X_test[california_housing.feature_names])

    X_validation["one_preds"] = tree_one.predict(X_validation[california_housing.feature_names])
    X_validation["two_preds"] = tree_two.predict(X_validation[california_housing.feature_names])
    results=[]
    ensemble = KWEnsembler(5)
    ensemble.fit(X_validation, y_validation)
    results.append(ensemble.predict(X_test.head(5),
                    california_housing.feature_names,
                    ["one_preds", "two_preds"]
                    ))

    print(np.mean(results-y_test.head(5).values))
    print(np.mean(np.mean(X_test["one_preds"])-y_test.head(5).values))
    print(np.mean(np.mean(X_test["two_preds"]) - y_test.head(5).values))