import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from ensemblem.model import KWEnsembler

class TestKWEnsembler(unittest.TestCase):
    def setup(self):
        #Mocking things that could be useful internally
        self.mock_dist_metric = MagicMock(return_value = np.array([1, 2, 3]))
        self.mock_weight_function = MagicMock(return_value = np.array([1, 0.5, 0.25]))

    @patch('sklearn.preprocessing.MinMaxScaler')
    def test_init(self, MockMinMaxScaler):
        ensembler = KWEnsembler(k=3,
                                bias=False,
                                dist_metric=self.mock_dist_metric)
        self.assertEqual(ensembler.k, 3)
        self.assertFalse(ensembler.bias)
        self.assertEqual(ensembler.dist_metric, self.mock_dist_metric)

    @patch('sklearn.preprocessing.MinMaxScaler')
    def test_fit(self, MockMinMaxScaler):
        ensembler = KWEnsembler(k=3, bias=False)

        X_neighbors = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_neighbors = pd.DataFrame({'target': [7, 8, 9]})
        features = ['feature1', 'feature2']

        ensembler.fit(X_neighbors, y_neighbors, features)

        # Ensure X_neighbors and y_neighbors are correctly set
        self.assertTrue((ensembler.X_neighbors.columns == features).all())
        self.assertTrue((ensembler.y_neighbors['target'] == y_neighbors['target']).all())

    def test_find_similar_neighbors(self):
        ensembler = KWEnsembler(k=1, bias=False)
        test_sample = pd.Series({'feature1': 5, 'feature2': 10})

        # Simulate a similar_space DataFrame
        similar_space = pd.DataFrame({
            'feature1': [1, 5, 10],
            'feature2': [2, 10, 20]
        }, index=['a', 'b', 'c'])

        neighbors = ensembler._find_similar_neighbors(test_sample, similar_space)

        # Check if the correct neighbors are identified
        # Assuming mock_dist_metric returns a fixed distance that implies 'a', 'b' are closest
        self.assertEqual(neighbors, ['b'])

    @patch('sklearn.preprocessing.MinMaxScaler')
    def test_predict(self, MockMinMaxScaler):

        ensembler = KWEnsembler(k=2, bias=True)

        self.mock_weight_function = MagicMock(return_value=np.array([1, 0.5, 0.25]))

        ensembler.weight_function = self.mock_weight_function

        # Setup mock data
        features = ['feature1', 'feature2']
        pred_columns = ['target']

        X_test = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        X_neighbors = pd.DataFrame({'feature1': [1, 5, 10], 'feature2': [2, 10, 20]})
        y_neighbors = pd.DataFrame({'target': [0.5, 1, 1.5]})

        ensembler.X_neighbors = X_neighbors
        ensembler.y_neighbors = y_neighbors

        ensembler.fit(X_neighbors, y_neighbors, features)
        predictions = ensembler.predict(X_test, features, features)

        # Assert predictions are made (mocked values would need calculating based on mock_weight_function)
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), len(X_test))


if __name__ == '__main__':
    unittest.main()

