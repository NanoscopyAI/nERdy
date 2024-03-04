"""junction analysis"""

import logging
import imageio
import numpy as np
from skimage.measure import regionprops

from junction_analysis_modules import JunctionAnalysisModules as JAM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JunctionAnalysis:
    """
    Analyzing junctions in a modality.

    Args:
        modality (str): The modality of the junctions.

    Attributes:
        modality (str): The modality of the junctions.
        junc_analysis_modules (JAM): An instance of the JAM class for junction analysis.

    """

    def __init__(self, modality):
        self.modality = modality
        self.junc_analysis_modules = JAM(modality)

    @staticmethod
    def get_std_img(path):
        """
        Load an image from the specified path and normalize its pixel values to the range [0, 1].

        Args:
            path (str): The path to the image file.

        Returns:
            ndarray: The normalized image.

        """
        # Load the image
        img = imageio.imread(path)

        if img.max() == img.min():
            return img

        # Normalize the pixel values to the range [0, 1]
        img_max = img.max()
        img_min = img.min()
        normalized_img = (img - img_min) / (img_max - img_min)

        return normalized_img

    @staticmethod
    def get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region):
        """
        Get junctions per connected component (CC) ID based on the specified region.
        For isolated junctions, only CCs with one junction are considered.
        For fuzzy junctions, only CCs with more than one junction are considered.
        
        Args:
            label_ids (dict): Dictionary with CC IDs as keys and junction coordinates as values.
            per_frame_junctions (list): List of all junction coordinates.
            labelled_img (ndarray): Labeled image with CC IDs.
            region (str): Region type ('iso' for isolated, 'fuz' for fuzzy).
            
        Returns:
            dict: Dictionary with CC IDs as keys and corresponding junction coordinates as values.
        """
        assert region in ['iso', 'fuz'], "Invalid region"
        
        # Get CC IDs with only one junction -> isolated junctions
        if region == 'iso':
            index_list = [idx for idx, junctions in label_ids.items() if idx != 0 and len(junctions) == 1]
        # Get CC IDs with more than one junction -> fuzzy junctions
        else:
            index_list = [idx for idx, junctions in label_ids.items() if idx != 0 and len(junctions) > 1]

        # Get junctions per CC ID
        label_id_junctions = {}
        for junction in per_frame_junctions:
            idx = labelled_img[tuple(junction)]
            if idx in index_list:
                label_id_junctions.setdefault(idx, set()).add(tuple(junction))

        return label_id_junctions

    def get_region_cc(self, group, series_num, region):
        """
        Get the connected components (CC) and labeled image for a specific region in a series.
        
        Args:
            group (str): Group name ('ATL', 'Climp', 'Control', 'RTN').
            series_num (int): Series number.
            region (str): Region type ('iso' for isolated, 'fuz' for fuzzy).
            
        Returns:
            tuple: A tuple containing the CC IDs for the specified region and the labeled image.
        """
        try:
            # Label the junctions and get relevant data
            ref_junctions, per_frame_junctions, labelled_img, ref_graph = self.junc_analysis_modules.label_junctions(group, series_num)

            # Separate the junctions into connected components (CCs)
            label_ids = self.junc_analysis_modules.separate_junc_cc(ref_junctions, labelled_img)

            # Get the junction areas for isolated and fuzzy regions
            iso, fuz = self.junc_analysis_modules.get_junction_areas(label_ids)

            if region == 'iso':
                cc_ids = self.get_cc_ids(labelled_img, iso)
            else:
                cc_ids = self.get_cc_ids(labelled_img, fuz)

            return cc_ids, labelled_img
        except Exception as e:
            # Handle exceptions gracefully
            raise ValueError(f"Error processing series {series_num} for group {group}: {e}")

    def cc_area_measure(self, group, region, rstart, rend):
        """
        Calculate the area of connected components (CC) in a given region and series range.

        Args:
            group (str): Group name ('ATL', 'Climp', 'Control', 'RTN').
            region (str): Region type ('iso' for isolated, 'fuz' for fuzzy).
            rstart (int): Starting series number.
            rend (int): Ending series number.

        Returns:
            list: A list of CC areas.

        Raises:
            Exception: If a series is not available.

        """
        cc_area_list = []

        for series_num in range(rstart, rend + 1):
            try:
                # Get the connected components (CC) and labeled image for the specified region and series
                region_cc, labelled_img = self.get_region_cc(group, series_num, region)
                regions = regionprops(labelled_img)

                if len(regions) == 0:
                    print("No connected components in series %d" % series_num)
                    continue

                # Calculate the area of each CC and add it to the list
                cc_areas = [regions[each - 1]['Area'] for each in region_cc]
                cc_area_list.extend(cc_areas)
            except Exception:
                print(f'Series {series_num} not available')
                continue

        # Save the CC area list to a pickle file
        # with open(f'cc_area_{group}_{region}.pkl', 'wb') as f:
        #     pickle.dump(cc_area_list, f)

        return cc_area_list

    @staticmethod
    def get_region_areas(label_id_junctions):
        """
        Calculate the areas of regions based on the number of junctions.

        Parameters:
        label_id_junctions (dict): A dictionary mapping label IDs to junctions.

        Returns:
        list: A list of region areas.
        """
        areas = []
        for id, juncs in label_id_junctions.items():
            areas.append(len(juncs))

        return areas

    @staticmethod
    def get_cc_ids(labelled_img, region):
        """
        Get connected component IDs for a specified region in a labelled image.

        Parameters:
        labelled_img (numpy.ndarray): The labelled image containing connected components.
        region (list): List of locations (x, y) within the region.

        Returns:
        list: List of connected component IDs for the specified region.
        """
        # Create a dictionary to store per component data
        cc_data = {cc_id: [] for cc_id in np.unique(labelled_img)}

        # Populate the dictionary with locations for the specified region
        for loc in region:
            loc_x, loc_y = loc[0], loc[1]
            cc_id = labelled_img[loc_x, loc_y]
            cc_data[cc_id].append(loc)

        # Extract CC ids for the specified region
        cc_ids = [cc_id for cc_id, data in cc_data.items() if cc_id > 0 and len(data) > 0]

        return cc_ids

    def get_region_areas_per_group(self, group, num_series, region):
        """
        Calculate the areas of a specific region for each series in a group.

        Args:
            group (str): The name of the group.
            num_series (int): The number of series in the group.
            region (str): The name of the region.

        Returns:
            list: A list of area values for each series.

        Raises:
            Exception: If a series is not available.
        """
        area_data = []
        
        for num in range(1, num_series + 1):
            try:
                ref_junctions, per_frame_junctions, labelled_img, ref_graph = self.junc_analysis_modules.label_junctions(group, num)
                label_ids, unassigned_cc_dict = self.junc_analysis_modules.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
                label_id_junctions = self.get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region)
                area_vals = self.get_region_areas(label_id_junctions)
                area_data.append(area_vals)
            except Exception as e:
                logger.error(f"Error processing series {num}: {e}")
                continue

        return area_data
