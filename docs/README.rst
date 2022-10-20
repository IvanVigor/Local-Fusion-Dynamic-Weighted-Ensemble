Dynamic Weighted Ensemble - Local Fusion [DOCUMENTATION UNDER REVIEW]
=====================================================================

This repository contain an implementation for the Dynamic Weighted
Ensemble (DWE) - Local Fusion method. Find the paper in this ref on
`IEEE <https://ieeexplore.ieee.org/document/8272838>`__.

Local Fusion is an ensemble techinque that could be used to improve
predictions by weighing appropriately the single models contribution.

Installation
------------

`Pypi <https://pypi.org/project/ensemblem/0.1/>`__



::

   pip install ensemblem==0.2.7

::



Usage
-----

First of all, you need to define the KWEnsembler class. Next, it's required to provide the search-space (it could be the validation set) in which the ensembler will find the nearest elemets to the generic test sample.

::

       from ensemblem.model import KWEnsembler
       ensemble = KWEnsembler(5)
       ensemble.fit(X_validation, y_validation)

::


Finally, calling the prediction method the class will produce the
forecasts.

::

     ensemble.predict(X_test,features_space,
     other_model_prediction_columns)

::

The method returns the prediction list in the same order in which they are provided. The class supports one or multiple samples to forecasts.

Example of using the KWEnsembler class
--------------------------------------

1. Load data (in this case we will use the california housing dataset). Refs to the dataset here: `California Housing <https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html>`__ 
::

california_housing = fetch_california_housing(as_frame=True)

::

2. Split data into train, validation and test sets. The validation dataset will be used in the following steps for k-nearest search.
::

X_train, y_train,X_validation, y_validation, X_test, y_test = split_sets(california_housing.frame.sample(frac=1), 0.70, 0.20, 0.10,
                                                                         california_housing.target_names[0])

::
3. Train multiple expert models on the train data. 

::

TreeRegressor_one = DecisionTreeRegressor(max_depth=3,
                                          random_state=123)
                    ...

TreeRegressor_one.fit(X_train, y_train)

::
4. Generate predictions for the test data
::

X_test["one_preds_rf"] = TreeRegressor_one.predict(X_test[california_housing.feature_names])

::
5. Train the ensembler on validation data
::

ensemble = KWEnsembler(50, bias=False)
ensemble.fit(X_validation, y_validation, california_housing.feature_names)



::
6. Generate predictions for the test dataset coming from the ensembler
::

results = ensemble.predict(X_test,
                california_housing.feature_names,
                ["one_preds_rf", "two_preds_rf",
                 "one_preds_r"],
                weight_function=w_inverse_log_LMAE)

::
7. Compare the predictions from the ensembler with the predictions from
   the expert models
::

print(metrics_table(y_test, X_test["one_preds_rf"], "Tree"))

::

Results & Benchmarks
--------------------

== =========== ============ ============ ============= ============
\  Model       MAPE         MAE          RMSE          RMSLE
== =========== ============ ============ ============= ============
0  **Esemble** **0.304129** **0.499381** **0.0016118** **0.211999**
1  Tree        0.370919     0.593606     0.00755926    0.249373
2  Tree        0.319638     0.511249     0.00224047    0.225012
3  RidgeCV     0.31537      0.531177     0.0131216     0.238018
== =========== ============ ============ ============= ============

Credits
------------

Algorithm Applications

-  Renewable energy forecasting - Wind
   `IEEE <https://ieeexplore.ieee.org/document/8272838>`__

-  An ensemble approach to sensor fault detection and signal
   reconstruction for nuclear system control
   `Elsevier <https://www.sciencedirect.com/science/article/pii/S0306454910000927>`__

Possible Improvements
---------------------

-  [Docs] General improvements over documentations

-  [Code] Clean-code

-  [Engineering] When dealing with features coming with magnitude and
   different meaning, it’s relevant to normalize values appropriately.

-  [Engineering] Search space without euclidean measure


