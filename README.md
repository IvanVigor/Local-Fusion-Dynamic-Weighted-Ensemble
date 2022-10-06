# Dynamic Weighted Ensemble - Local Fusion

This repository contain an implementation for the Dynamic Weighted Ensemble (DWE) - Local Fusion method. Find the paper in this ref on [IEEE](https://ieeexplore.ieee.org/document/8272838).

Local Fusion is an ensemble techinque, which can be used to improve predictions by weighing appropriately the single models contribution.

![Arch](https://github.com/IvanVigor/IvanVigor.github.io/blob/main/imgs/dwe.png?raw=true)

## Installation

TODO


## Usage

Let's define the KWEnsembler class. And then define the feature space in which the neighbours should be found.


	ensemble = KWEnsembler(5)
	ensemble.fit(X_validation, y_validation)


Finally, calling the prediction method the class will produce the forecasts.

	ensemble.predict(X_test,
                    features_space,
                    other_model_prediction_columns)

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



## Applications

Algorithm Applications

- Renewable energy forecasting - Wind [IEEE](https://ieeexplore.ieee.org/document/8272838)

- An ensemble approach to sensor fault detection and signal reconstruction for nuclear system control [Elsevier](https://www.sciencedirect.com/science/article/pii/S0306454910000927)


## Possible Improvements

-  When dealing with features coming with magnitude and different meaning, it's relevant to normalize values appropriately.

- Search space without euclidean measure


## Licence
The code is provided with a MIT licence. 

