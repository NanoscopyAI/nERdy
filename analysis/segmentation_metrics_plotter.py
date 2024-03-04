"""segmentation metrics plotter"""

import pickle as pkl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# path = 'pickle_files/segmentation_measures/'

class SegmentationMetricsPlotter:
    """
    Plot the segmentation performance of the different methods.

    Attributes:
        modality (list): The modalities of the images.
        metrics (list): The segmentation metrics.
        methods (list): The segmentation methods.
    """

    def __init__(self):
        self.modality = ['confocal', 'sted']
        self.metrics = ['dice', 'f1', 'iou']
        self.methods = ['AnalyzER', 'ERnet', 'ERnet-v2', 'nERdy', 'nERdy+']

    def load_data(self, modality, metric):
        """
        Load data from a pickle file based on the specified modality and metric.

        Parameters:
        modality (str): The modality of the data.
        metric (str): The metric of the data.

        Returns:
        data: The loaded data from the pickle file.
        """
        assert modality in self.modality, 'Invalid modality'
        assert metric in self.metrics, 'Invalid metric'
        return pkl.load(open(f'pickle_files/segmentation_measures/{modality}_{metric}.pkl', 'rb'))

    @staticmethod
    def get_series(data):
        """
        Flatten a nested list into a single list.

        Args:
            data (list): A nested list containing the data.

        Returns:
            list: A flattened list.
        """
        return [value for sublist in data for value in sublist]

    def get_segmentation_perf(self):
        """
            Calculate and plot the segmentation performance metrics.

            Returns:
                None
        """
        conf_dice = self.load_data('confocal', 'dice')
        conf_f1 = self.load_data('confocal', 'f1')
        conf_jaccard = self.load_data('confocal', 'iou')

        sted_dice = self.load_data('sted', 'dice')
        sted_f1 = self.load_data('sted', 'f1')
        sted_jaccard = self.load_data('sted', 'iou')

        modality_data = {
            'Confocal': conf_dice + conf_f1 + conf_jaccard,
            'STED': sted_dice + sted_f1 + sted_jaccard
        }

        df = pd.DataFrame()

        val_data = conf_dice + conf_f1 + conf_jaccard + sted_dice + sted_f1 + sted_jaccard

        vals = self.get_series(val_data)

        df['Values'] = vals

        metric_names = []
        method_names = []
        col_name = []

        metric_names = []

        for method in conf_dice:
            metric_names.append(['Dice score']*len(method))
        for method in conf_f1:
            metric_names.append(['F1 score']*len(method))
        for method in conf_jaccard:
            metric_names.append(['Jaccard Index']*len(method))
        for method in sted_dice:
            metric_names.append(['Dice score']*len(method))
        for method in sted_f1:
            metric_names.append(['F1 score']*len(method))
        for method in sted_jaccard:
            metric_names.append(['Jaccard Index']*len(method))

        for num, method_name in enumerate(self.methods):
            method_names.append([f'{method_name}']*len(conf_dice[num]))
        for num, method_name in enumerate(self.methods):
            method_names.append([f'{method_name}']*len(conf_f1[num]))
        for num, method_name in enumerate(self.methods):
            method_names.append([f'{method_name}']*len(conf_jaccard[num]))    
        for num, method_name in enumerate(self.methods):
            method_names.append([f'{method_name}']*len(sted_dice[num]))
        for num, method_name in enumerate(self.methods):
            method_names.append([f'{method_name}']*len(sted_f1[num]))
        for num, method_name in enumerate(self.methods):
            method_names.append([f'{method_name}']*len(sted_jaccard[num]))

        for modality in modality_data:
            total_lengths = [sum(len(data) for data in modality_data[modality])]
            col_name.extend([modality] * total_length for total_length in total_lengths)

        metric_series = self.get_series(metric_names)
        method_series = self.get_series(method_names)
        col_series = self.get_series(col_name)

        df['Metric'] = metric_series
        df['Method'] = method_series
        df['Modality'] = col_series

        ax = sns.catplot(kind='box', x="Metric", y="Values", hue="Method", col='Modality', data=df, showfliers=False, width=0.8, palette="Set2", legend=False)

        for a in ax.axes[0]:
            for i, metric in enumerate(df['Metric'].unique()):
                a.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)

        plt.ylim(0.2, 1.0)
        ax.set_xticklabels(['Dice', 'F1', 'IoU'], fontsize=8.5, weight='bold')
        ax.set_yticklabels([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=8.5, weight='bold')

        ax.set_xlabels('')
        ax.set_ylabels('')

        plt.gcf().set_size_inches(4.5, 3.5)

        plt.show()
        # plt.savefig('combined_segmentation_plots.png', dpi=300, bbox_inches='tight', pad_inches=0.01)

        # plt.close()

# segplotter = SegmentationMetricsPlotter()
# segplotter.get_segmentation_perf()
