# Dynamic-Weighted-Ensemble

This repository contain an implementation for the Dynamic Weighted Ensemble (DWE). Find the paper in this ref on [IEEE](https://ieeexplore.ieee.org/document/8272838). DWE is an ensemble techinque that could be used to improve predictions by weighing appropriately the single models contribution.

## Installation

ToUpload


## Usage

Let's define the KWEnsembler class. And then define the feature space in which the neighbours should be found.


	ensemble = KWEnsembler(5)
	ensemble.fit(X_validation, y_validation)


Finally, calling the prediction method the class will produce the forecasts.

	ensemble.predict(X_test,
                    features_space,
                    other_model_prediction_columns)



## Possible Improvements

-  When dealing with features coming with magnitude and different meaning, it's relevant to normalize values appropriately.
- 


## Licence
The software is provided with a MIT licence. 

