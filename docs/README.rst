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

Let's define the KWEnsembler class. And then define the feature space in which the neighbours should be found.



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


The method returns the prediction list in the same order in which they
are provided. The class supports one or multiple samples to forecats.

Example of using the KWEnsembler class
--------------------------------------

1. Load data
2. Split data into train, validation and test sets
3. Train multiple expert models on the train data
4. Generate predictions for the test data
5. Train the ensembler on validation data
6. Generate predictions for the test dataset coming from the ensembler
7. Compare the predictions from the ensembler with the predictions from
   the expert models

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

-  [Docs] General improveiments over documentions

-  [Code] Clean-code

-  [Engineering] When dealing with features coming with magnitude and
   different meaning, it’s relevant to normalize values appropriately.

-  [Engineering] Search space without euclidean measure

