import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Adjust path to include project root
from analysis.segmentation_metrics_plotter import SegmentationMetricsPlotter

class TestSegmentationMetricsPlotter(unittest.TestCase):

    def setUp(self):
        self.plotter = SegmentationMetricsPlotter()

    def test_load_data(self):
        data = self.plotter.load_data('confocal', 'dice')
        self.assertIsInstance(data, list)
        # Add more specific checks based on your expected data structure

    def test_get_series(self):
        data = [[1, 2], [3, 4]]
        flattened = self.plotter.get_series(data)
        self.assertEqual(flattened, [1, 2, 3, 4])

    def test_get_segmentation_perf(self):
        # Mock or use sample data to test the plotting function
        self.plotter.get_segmentation_perf()
        # Optionally, you can save the plot and check if it was created
        # For example, uncomment these lines to save and check the plot file
        # plt.savefig('test_segmentation_plot.png')
        # self.assertTrue(os.path.isfile('test_segmentation_plot.png'))

if __name__ == '__main__':
    unittest.main()

