import unittest
from unittest.mock import patch, mock_open
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.graph_metrics_plotter import GraphMetricsPlotter

class TestGraphMetricsPlotter(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data=b'data')
    @patch('pickle.load')
    def test_load_pickle(self, mock_pkl_load, mock_open):

        # Test loading pickle file
        mock_pkl_load.return_value = [1, 2, 3]
        plotter = GraphMetricsPlotter('confocal')
        data = plotter.load_pickle('AnalyzER')
        self.assertTrue(np.array_equal(data, np.array([1, 2, 3])))

        # Assert that the correct file is opened
        mock_open.assert_called_once_with('analysis/pickle_files/graph_measures/confocal_AnalyzER_graph_err.pkl', 'rb')

    def test_std_data(self):
        plotter = GraphMetricsPlotter('confocal')
        data = np.array([1, 2, 3, 4, 5])
        standardized_data = plotter.std_data(data)
        self.assertEqual(standardized_data, [0.0, 0.25, 0.5, 0.75, 1.0])

    @patch.object(GraphMetricsPlotter, 'load_pickle')
    def test_get_graph_features(self, mock_load_pickle):
        plotter = GraphMetricsPlotter('confocal')
        mock_data = np.array([list(range(21)) for _ in plotter.features])
        mock_load_pickle.return_value = mock_data
        features_data = plotter.get_graph_features('AnalyzER')
        expected_keys = ['num_nodes', 'num_edges', 'assortativity', 'clustering', 'num_components', 'ratio_nodes', 'ratio_edges', 'global_efficiency', 'density']
        for key in expected_keys:
            self.assertIn(key, features_data)

            # Assert that the length of the data is 21
            self.assertEqual(len(features_data[key]), 21)

    def test_prepare_data(self):
        plotter = GraphMetricsPlotter('confocal')
        dummy_data = {
            'AnalyzER': {
                'num_nodes': [0.1, 0.2, 0.3],
                'num_edges': [0.4, 0.5, 0.6],
                'assortativity': [0.7, 0.8, 0.9],
                'clustering': [1.0, 1.1, 1.2],
                'num_components': [1.3, 1.4, 1.5],
                'ratio_nodes': [1.6, 1.7, 1.8],
                'ratio_edges': [1.9, 2.0, 2.1],
                'global_efficiency': [2.2, 2.3, 2.4],
                'density': [2.5, 2.6, 2.7]
            },
            'ERnet': {
                'num_nodes': [2.8, 2.9, 3.0],
                'num_edges': [3.1, 3.2, 3.3],
                'assortativity': [3.4, 3.5, 3.6],
                'clustering': [3.7, 3.8, 3.9],
                'num_components': [4.0, 4.1, 4.2],
                'ratio_nodes': [4.3, 4.4, 4.5],
                'ratio_edges': [4.6, 4.7, 4.8],
                'global_efficiency': [4.9, 5.0, 5.1],
                'density': [5.2, 5.3, 5.4]
            }
        }
        df = plotter.prepare_data(dummy_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 54)  # 3 features * 9 metrics * 2 methods

    @patch('matplotlib.pyplot.show')
    def test_plot(self, mock_show):
        plotter = GraphMetricsPlotter('confocal')
        with patch('matplotlib.pyplot.savefig'):
            plotter.plot()
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
