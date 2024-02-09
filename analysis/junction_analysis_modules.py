import numpy as np
from skimage.measure import label, regionprops
import itertools
import sknw
import imageio
from plantcv import plantcv as pcv
import copy
import graph_connector_modules as gcm
import matplotlib.pyplot as plt
import os


home = os.path.expanduser('~')

confocal_data_path = f'{home}/MIAL/data/confocal-data/'
sted_data_path = f'{home}/MIAL/data/sted-data/'


class JunctionAnalysisModules:

    def __init__(self, modality):
        self.data_path = sted_data_path if modality == 'sted' else confocal_data_path

    def get_skeleton(self, img_path):
        """
        @param img_path: path to image
        @return: skeleton (ndarray) - skeleton of the image
        """
        img = imageio.imread(img_path)
        return pcv.morphology.skeletonize(mask=img)

    def skel_to_graph(self, skel_img_path):
        """
        @param skel_img_path:
        @return: graph (Graph) - graph of the skeleton
        """

        skel = self.get_skeleton(skel_img_path)
        return sknw.build_sknw(skel, multi=True, iso=False)

    def get_junctions(self, path_skel):
        """
        @param path_er: path to er image
        @param path_skel: path to skeleton image
        @return: junctions (list) - provides all junctions with degree > 2 from the mean projection proc skeleton
        """

        graph = self.skel_to_graph(path_skel)

        node_set, degree_list = graph.nodes, graph.degree

        node_coords = np.array([node_set[node]['o'] for node in node_set])
    
        return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]

    def get_all_junc(self, group, num_series):
        # get reference junctions based on mean projection frame and per frame junctions for each series, all groups
        """

        @param group: group to be analyzed
        @param num_series: sequence number
        @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, per_frame_junctions (list) - provides all junctions per skel frame
        """

        group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

        mean_skel = f'{self.data_path}{group}/er_mean_proc/{group.lower()}{num_series}_proc_skel.png'

        ref_junctions = self.get_junctions(mean_skel)
        ref_junctions = [[each[0], each[1]] for each in ref_junctions]

        per_frame_junctions = []
        
        # for frame in range(fr_start, fr_end):
        for frame in range(100):
            skeleton_path = f'{self.data_path}{group}/skel/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

            junctions = self.get_junctions(skeleton_path)

            junc_array = [[junc[0], junc[1]] for junc in junctions]
            per_frame_junctions.extend(junc_array)

        return ref_junctions, per_frame_junctions

    def label_junctions(self, group, series_num):
        """
        @param group: group to be analyzed
        @param series_num: sequence number
        @return: ref_junctions (ndarray) - provides all reference junctions, per_frame_junctions (ndarray) - provides all junctions per skel frame, labelled_img (ndarray) - labelled image of all junctions
        """

        ref_junctions, per_frame_junctions = self.get_all_junc(group, series_num)

        ref_junctions = np.array(ref_junctions)
        per_frame_junctions = np.array(per_frame_junctions)

        spread_img = np.zeros((128, 128))
        for each in per_frame_junctions:
            spread_img[each[0], each[1]] = 255.

        labelled_img = label(spread_img, connectivity=2)

        return ref_junctions, per_frame_junctions, labelled_img

    def get_ref_junc_per_CC_id(self, reference_junctions, connected_components):
        # get CC_id and corresponding junctions, get CC_ids with at
        """
        Return the reference junctions per connected component and the list of connected components with at least 1 reference
        junction.

        :param reference_junctions: (ndarray) the reference junctions
        :param connected_components: (ndarray) the connected components for the junctions
        :return: label_values (dict) provides corresponding reference junctions per cc_id,
                 assigned_components (list) provides cc with at least 1 reference junction
        """
        label_ids = {}
        assigned_components = []

        for junction in reference_junctions:
            if connected_components[junction[0], junction[1]] != 0:
                cc_id = connected_components[junction[0], junction[1]]
                if cc_id not in label_ids:
                    label_ids[cc_id] = []
                label_ids[cc_id].append([junction[0], junction[1]])
                assigned_components.append(cc_id)

        return label_ids, assigned_components

    def get_uncertain_junctions(self, labelled_img, per_frame_junctions, num_components, assigned_components):
        """
        Return junction CCs without a reference junction

        :param labelled_img: (ndarray) Image with all labelled CCs
        :param per_frame_junctions: (ndarray) junction per frame
        :param num_components: (int) number of unique CCs
        :param assigned_components: (list) CCs with at least 1 ref junction
        :return: unassigned_cc_dict (dict), CC label id and per frame junctions for CCs without reference junction
        """
        # list of CCs without a ref junction
        unassigned_components = [x for x in num_components if x not in assigned_components]

        unassigned_cc_dict = {}

        # Get junctions per frame for CCs without ref junction
        for each in per_frame_junctions:
            cc_label = labelled_img[each[0], each[1]]
            if cc_label != 0 and cc_label in unassigned_components:
                if cc_label not in unassigned_cc_dict.keys():
                    unassigned_cc_dict[(labelled_img[each[0], each[1]])] = []
                unassigned_cc_dict[(labelled_img[each[0], each[1]])].append([each[0], each[1]])
        return unassigned_cc_dict

    def separate_junc_cc(self, ref_junctions, per_frame_junctions, labelled_img):
        """
        Return ref junctions per CC and CCs without reference junction dicts.

        """
        regions = regionprops(labelled_img)

        num_components = np.unique(labelled_img)

        label_ids, assigned_components = self.get_ref_junc_per_CC_id(ref_junctions, labelled_img)
        # print(label_vals)

        unassigned_cc_dict = self.get_uncertain_junctions(labelled_img, per_frame_junctions, num_components, assigned_components)

        # return label_vals, cc_area_dict, unassigned_cc_dict
        return label_ids, unassigned_cc_dict

    def get_junction_areas(self, label_ids, unassigned_cc_dict):
        """
        Returns junctions arrays for iso, fuz and unknown classes
        """

        isolated_junctions = []
        fuzzy_junctions = []
        unknown_junctions = []

        for cc_id, junctions in label_ids.items():
            if cc_id != 0:
                if len(junctions) == 1:
                    isolated_junctions.append(junctions[0])
                    #isolated_junc_area.append(cc_area_dict[k])
                else:
                    fuzzy_junctions.append(junctions)

        for junctions in unassigned_cc_dict.values():
            unknown_junctions.extend(junctions)

        isolated_junctions = np.array(isolated_junctions)

        fuzzy_junctions = list(itertools.chain.from_iterable(fuzzy_junctions))
        fuzzy_junctions = np.array(fuzzy_junctions)

        unknown_junctions = list(itertools.chain.from_iterable(unknown_junctions))
        unknown_junctions = np.array(unknown_junctions)

        return isolated_junctions, fuzzy_junctions, unknown_junctions
