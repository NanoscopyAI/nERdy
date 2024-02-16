# Relative errors are stored in pickle files. 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


class GraphMetricsPlotter:
    def __init__(self, modality):
        self.features = ['num_nodes', 'num_edges', 'assortativity', 'clustering', 'num_components', 'ratio_nodes', 'ratio_edges', 'global_efficiency', 'density']
        self.modality = modality

    def std_data(self, data):
        # Standardize data for same scale
        return list((data - min(data)) / (max(data) - min(data)))
    
    def load_pickle(self, modality, method):
        return np.array(pkl.load(open(f'{modality}_{method}_graph_err.pkl', 'rb')))
    
    def get_graph_features(self, modality, method):
        method_data = {}
        for feature in self.features:
            data = self.load_pickle(modality, method)[self.features.index(feature)]
            method_data[feature] = self.std_data(data)
        return method_data
    
    def plot(self):
        analyzer_data = self.get_graph_features(self.modality, 'analyzer')
        ernet_data = self.get_graph_features(self.modality, 'ernet')
        ernet_v2_data = self.get_graph_features(self.modality, 'erv2')
        nerdy_data = self.get_graph_features(self.modality, 'nerdy')
        nerdy_p4m_data = self.get_graph_features(self.modality, 'p4m')
        
        df = pd.DataFrame()
        
        val_data = []
        metric_names = []
        method_names = []
        
        for feature in self.features:
            val_data.extend(analyzer_data[feature] + ernet_data[feature] + ernet_v2_data[feature] + nerdy_data[feature] + nerdy_p4m_data[feature])
        
            metric_names.append([feature]*len(analyzer_data[feature]))
            metric_names.append([feature]*len(ernet_data[feature]))
            metric_names.append([feature]*len(ernet_v2_data[feature]))
            metric_names.append([feature]*len(nerdy_data[feature]))
            metric_names.append([feature]*len(nerdy_p4m_data[feature]))
        
            method_names.append(['AnalyzER']*len(analyzer_data[feature]))
            method_names.append(['ERnet']*len(ernet_data[feature]))
            method_names.append(['ERnet-v2']*len(ernet_v2_data[feature]))
            method_names.append(['nERdy']*len(nerdy_data[feature]))
            method_names.append(['nERdy+']*len(nerdy_p4m_data[feature]))
        
        df['Values'] = val_data
        df['Metric'] = [item for sublist in metric_names for item in sublist]
        df['Method'] = [item for sublist in method_names for item in sublist]
        
        ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.6, width=0.7, palette="Set2")
        
        yt = ax.get_yticks()
        yt = [f'{y:.1f}' for y in yt]
        ax.set_yticklabels(yt, fontsize=8)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fancybox=True, shadow=True, fontsize=6)
        
        ax.set_xticks(np.arange(0, 9, 1), labels=['NN', 'NE', 'AS', 'CL', 'NC', 'RN', 'RE', 'GE', 'D'], minor=False, linespacing=3.5, fontsize=8)

        plt.xlabel('Graph Property', fontsize=9)
        plt.ylabel('Relative Error (normalized)', fontsize=9)

        plt.gcf().set_size_inches(6, 3) # new size

        plt.show()

