import numpy as np
from skimage.measure import label, regionprops
import itertools
import sknw
import imageio
from plantcv import plantcv as pcv
import copy
import matplotlib.pyplot as plt
import skimage
import os

confocal_data_path = '/path/to/confocal/data'
sted_data_path = '/path/to/sted/data'

class JunctionAnalysisModules:

    def __init__(self, modality):
        self.data_path = sted_data_path if modality == 'sted' else confocal_data_path

    def get_skeleton(self, img_path):
        """
        Get the skeleton of an image.

        Args:
            img_path (str): The path to the image file.

        Returns:
            numpy.ndarray: The skeletonized image.

        """
        img = imageio.imread(img_path)
        return pcv.morphology.skeletonize(mask=img)

    def skel_to_graph(self, skel_img_path):
        """
        Converts a skeleton image to a graph representation.

        Args:
            skel_img_path (str): The file path of the skeleton image.

        Returns:
            sknw.graph.Graph: The graph representation of the skeleton image.
        """
        skel = imageio.imread(skel_img_path)
        return sknw.build_sknw(skel, multi=False, iso=False)

    def get_junctions(self, path_skel):
        """
        Get the coordinates of junctions in a graph.

        Args:
            path_skel (str): The file path to the skeleton file.

        Returns:
            list: A list of coordinates of junctions with degree greater than 2.
        """
        graph = self.skel_to_graph(path_skel)

        node_set, degree_list = graph.nodes, graph.degree

        # get node coordinates
        node_coords = np.array([node_set[node]['o'] for node in node_set])

        # return coordinates of junctions with degree > 2    
        return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]

    def get_all_junc(self, group, num_series):
            """
            Get reference junctions based on mean projection frame and per frame junctions for each series, all groups.

            Args:
                group (str): The group name.
                num_series (int): The series number.

            Returns:
                tuple: A tuple containing the reference junctions, per frame junctions, and reference graph.
            """
            group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

            if self.data_path == sted_data_path:
                mean_skel_path = f'{self.data_path}/{group.lower()}/gt_skel/sted_{group.lower()}{num_series}_proc_skel.png'
            else:
                mean_skel_path = f'{self.data_path}/{group.lower()}/gt_skel/conf_{group.lower()}{num_series}_proc_skel.png'

            if os.path.exists(mean_skel_path):
                ref_graph = self.skel_to_graph(mean_skel_path)

                ref_junctions = self.get_junctions(mean_skel_path)
                ref_junctions = [[each[0], each[1]] for each in ref_junctions]

                per_frame_junctions = []
                
                for frame in range(100):
                    skeleton_path = f'{self.data_path}/vess_enh_unet/{group.lower()}/skel/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

                    junctions = self.get_junctions(skeleton_path)

                    junc_array = [[junc[0], junc[1]] for junc in junctions]
                    per_frame_junctions.extend(junc_array)
            
            else:
                ref_junctions = []
                per_frame_junctions = []
                ref_graph = []

            return ref_junctions, per_frame_junctions, ref_graph
    

    def label_junctions(self, group, series_num):
            """
            Labels the junctions in the given group and series number.

            Args:
                group (str): The group identifier.
                series_num (int): The series number.

            Returns:
                tuple: A tuple containing the following elements:
                    - ref_junctions (list): The reference junctions.
                    - per_frame_junctions (list): The per-frame junctions.
                    - labelled_img (ndarray): The labelled image of the spread of junctions.
                    - ref_graph (Graph): The reference graph.
            """
            
            ref_junctions, per_frame_junctions, ref_graph = self.get_all_junc(group, series_num)

            # Check if reference junctions exist
            if not ref_junctions:
                return [], [], [], []

            ref_junctions = np.array(ref_junctions)
            per_frame_junctions = np.array(per_frame_junctions)

            # Create an image to label the spread of junctions
            junc_spread_img = np.zeros((128, 128))

            # Set pixel values for per-frame junctions
            for junction in per_frame_junctions:
                junc_spread_img[junction[0], junction[1]] = 255.

            # Label connected components in the spread image
            labelled_img = label(junc_spread_img, connectivity=2)

            return ref_junctions, per_frame_junctions, labelled_img, ref_graph


    def get_ref_junc_per_CC_id(self, reference_junctions, connected_components):
        """
        Get reference junctions per connected component ID.

        Args:
            reference_junctions (list): List of reference junctions.
            connected_components (numpy.ndarray): Array representing connected components.

        Returns:
            dict: A dictionary where the keys are connected component IDs and the values are lists of reference junctions belonging to each connected component.
        """

        label_ids = {}

        for junction in reference_junctions:
            if connected_components[junction[0], junction[1]] != 0:
                cc_id = connected_components[junction[0], junction[1]]
                if cc_id not in label_ids:
                    label_ids[cc_id] = []
                label_ids[cc_id].append([junction[0], junction[1]])

        return label_ids

    def get_uncertain_junctions(self, labelled_img, per_frame_junctions, num_components, assigned_components):
            """
            Get the uncertain junctions for connected components without a reference junction.

            Args:
                labelled_img (numpy.ndarray): The labelled image.
                per_frame_junctions (list): List of junctions per frame.
                num_components (list): List of all connected components.
                assigned_components (list): List of connected components with a reference junction.

            Returns:
                dict: A dictionary containing the uncertain junctions per connected component.
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

    def separate_junc_cc(self, ref_junctions, labelled_img):
        """
        Separates the junctions based on connected components (CC) in the labelled image.

        Args:
            ref_junctions (list): List of reference junctions.
            labelled_img (ndarray): Labelled image containing connected components.

        Returns:
            list: List of label IDs corresponding to the reference junctions.
        """
        label_ids = self.get_ref_junc_per_CC_id(ref_junctions, labelled_img)

        return label_ids

    def get_junction_areas(self, label_ids):
        """
        Get the areas of isolated and fuzzy junctions.

        Parameters:
        - label_ids (dict): A dictionary containing label IDs and corresponding junctions.

        Returns:
        - isolated_junctions (numpy.ndarray): An array of isolated junctions.
        - fuzzy_junctions (numpy.ndarray): An array of fuzzy junctions.
        """
        isolated_junctions = []
        fuzzy_junctions = []

        for cc_id, junctions in label_ids.items():
            if cc_id != 0:
                if len(junctions) == 1:
                    isolated_junctions.append(junctions[0])
                else:
                    fuzzy_junctions.append(junctions)

        isolated_junctions = np.array(isolated_junctions)

        fuzzy_junctions = list(itertools.chain.from_iterable(fuzzy_junctions))
        fuzzy_junctions = np.array(fuzzy_junctions)

        return isolated_junctions, fuzzy_junctions
