import unittest
from ensemblem.model import KWEnsembler

# ToDo: remove this bad data retrieval
from sklearn.datasets import fetch_california_housing


class MyTestCase(unittest.TestCase):
    def test_fit_method(self):
        california_housing = fetch_california_housing(as_frame=True)
        ensemble = KWEnsembler(50, bias=False)
        ensemble.fit(
            california_housing.frame,
            california_housing.target,
            california_housing.feature_names,
        )
        self.assertTrue(ensemble.X_val.shape[0] == california_housing.frame.shape[0])
        self.assertTrue(ensemble.y_val.shape[0] == california_housing.target.shape[0])

    def test_fit(self):
        ensemble = KWEnsembler(50, bias=False)
        self.assertTrue(ensemble.k == 50)


# ToDo: remove this bad data retrieval


if __name__ == "__main__":
    unittest.main()
