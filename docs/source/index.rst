.. KWEnsembler documentation master file, created by
   sphinx-quickstart on Sun Oct 16 17:44:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KWEnsembler's documentation
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ensemblem


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Introduction [WORKING ON]
=========================

# Dynamic Weighted Ensemble - Local Fusion

This repository contain an implementation for the Dynamic Weighted Ensemble (DWE) - Local Fusion method. 

Local Fusion is an ensemble techinque that could be used to improve predictions by weighing appropriately the single models contribution.

<img src="https://github.com/IvanVigor/IvanVigor.github.io/blob/main/imgs/dwe.png?raw=true)" width="900" height="500"/>

## Installation

[Pypi](https://pypi.org/project/ensemblem/0.1/)

```{r setup, include=FALSE}

   pip install ensemblem==0.2.7

```

## Usage

Let's define the KWEnsembler class. And then define the feature space in which the neighbours should be found.


```{r setup, include=FALSE}
   from ensemblem.model import KWEnsembler
   ensemble = KWEnsembler(5)
   ensemble.fit(X_validation, y_validation)
```

Finally, calling the prediction method the class will produce the forecasts.

```{r setup, include=FALSE}
   ensemble.predict(X_test,
                    features_space,
                    other_model_prediction_columns)
```

The method returns the prediction list in the same order in which they are provided. The class supports one or multiple samples to forecats.



## Example of using the KWEnsembler class


1. Load data
2. Split data into train, validation and test sets
3. Train multiple expert models on the train data
4. Generate predictions for the test data
5. Train the ensembler on validation data
6. Generate predictions for the test dataset coming from the ensembler
7. Compare the predictions from the ensembler with the predictions from the expert models


## Results & Benchmarks

|    | Model   |     MAPE |      MAE |      RMSE |    RMSLE |
|---:|:--------|---------:|---------:|----------:|---------:|
|  0 | ***Esemble*** | ***0.304129*** | ***0.499381*** | ***0.0016118*** | ***0.211999*** |
|  1 | Tree    | 0.370919 | 0.593606 | 0.00755926 | 0.249373 |
|  2 | Tree    | 0.319638 | 0.511249 | 0.00224047 | 0.225012 |
|  3 | RidgeCV | 0.31537 | 0.531177 | 0.0131216 | 0.238018 |



## Credits

Algorithm Applications

- Renewable energy forecasting - Wind [IEEE](https://ieeexplore.ieee.org/document/8272838)

- An ensemble approach to sensor fault detection and signal reconstruction for nuclear system control [Elsevier](https://www.sciencedirect.com/science/article/pii/S0306454910000927)






