import imageio
import skimage
from skimage.filters import threshold_otsu, threshold_local
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import seaborn as sns
import os
from sklearn.feature_selection import f_oneway
import sknw
import networkx as nx
import numpy as np
from skimage import metrics
import pandas as pd
import scipy
import pickle
from scipy import spatial
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage import measure

from junction_analysis_modules import JunctionAnalysisModules as JAM

max_val = 999
confocal_data_path = '/localhome/asa420/MIAL/data/confocal-data/'
sted_data_path = '/localhome/asa420/MIAL/data/sted-data/'
# junc_analysis = JAM('sted')
junc_analysis = JAM('confocal')



def get_std_img(path):
    """
    Get standardised image
    @param path: path to image
    @return: standardised image
    """
    img = imageio.imread(path)
    if img.max() == img.min():
        return img
    img_max = img.max()
    img_min = img.min()
    return (img - img_min) / (img_max - img_min)


# patch1_vals -> list of intensity values per patch

def get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region):
    """
    Get junctions per CC id
    @param label_ids: dict of CC ids and corresponding junctions
    @param per_frame_junctions: list of junctions per frame
    @param labelled_img: labelled image
    @param region: 'iso' or 'non-iso'
    @return: dict of CC ids and junctions
    """
    # get CC ids with only one junction -> isolated junctions
    if region == 'iso':
        index_list = [idx for idx, junctions in label_ids.items() if idx != 0 and len(junctions) == 1]
    # get CC ids with more than one junction -> fuzzy junctions
    else:
        index_list = [idx for idx, junctions in label_ids.items() if idx != 0 and len(junctions) > 1]

    # get junctions per CC id
    label_id_junctions = {}
    for junction in per_frame_junctions:
        idx = labelled_img[tuple(junction)]
        if idx in index_list:
            label_id_junctions.setdefault(idx, set()).add(tuple(junction))

    return label_id_junctions


# ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions('ATL', 1)
# label_ids, assigned_components = junc_analysis.get_ref_junc_per_CC_id(ref_junctions, labelled_img)
# label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, 'fuz')


# get pixel data per CC id
def get_CC_patch_data(cc_id, group, num, channel):
    """
    Get pixel data per CC id

    modalities: confocal or sted
    junction: junction
    group: 'ATL', 'Climp', 'Control', 'RTN'
    num: number of series

    returns list of lists of pixel values
    """
    
    group_prefixes = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
    pixel_data = []
    
    ch_id = 0 if channel == 'egfp' else 1
    
    for frame in range(100):
        file_name = f'{group_prefixes[group]}{num}_decon_t0{frame:02d}_ch0{ch_id}_std.png'
        
        if channel == 'egfp':
            file_path = os.path.join(confocal_data_path, group, 'std', file_name)
        else:
            file_path = os.path.join(confocal_data_path, group, 'std_mch', file_name)
        img = imageio.imread(file_path)
        pixel_data.append(img[cc_id])
    return pixel_data


def get_sequence_CC_pixel_data(label_id_junctions, group, num, channel):
    """
    Get pixel data for all CC id in a sequence

    @param label_id_junctions: dict of CC ids and junctions
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num: number of series
    @return: list of lists of lists of lists of pixel values
    """

    #Get the patch data for all junctions in the image
    sequence_data = []

    for idx, junctions in label_id_junctions.items():
        junctions = list(junctions)
        for junction in junctions:
            pixel_data = get_CC_patch_data(junction, group, num, channel)
            sequence_data.append(pixel_data)
    return sequence_data


# def get_per_CC_pixel_data(group, num_series, region, channel):
def get_group_CC_pixel_data(group, num_series, region, channel):
    """
    Get CC intensity variation for all CCs across sequences in a group
    
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num_series: number of series
    @param channel: 'egfp' or 'mch'
    @return: list of lists of lists of lists of pixel values
    """

    group_data = []

    for num in range(1, num_series + 1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, assigned_components = junc_analysis.get_ref_junc_per_CC_id(ref_junctions, labelled_img)
        label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region)

        sequence_data = get_sequence_CC_pixel_data(label_id_junctions, group, num, channel)
        group_data.append(sequence_data)
    return group_data


def get_mean_std_per_CC_pixel_data(data, measure):
    """
    Get mean and std of pixel data per CC id
    @param data: list of lists of lists of lists of pixel values
    @return: list of means and list of stds
    """
    if data:
        measure_data = []
        for seq_num in data:
            for junction_data in seq_num:
                if measure == 'mean':
                    measure_data.append(np.mean(np.array(junction_data, dtype=np.float)/255.))
                elif measure == 'std':
                    measure_data.append(np.std(np.array(junction_data, dtype=np.float)/255.))
        return measure_data


def create_per_CC_pixel_data_pickles(group, num_series, region, channel):
    """
    Create pickles of pixel data per CC id
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num_series: number of sequences per group
    @param region: 'isolated' or 'fuzzy'
    """
    data = get_group_CC_pixel_data(group, num_series, region, channel)
    
    pickle.dump(data, open(f'{group}_{channel}_{region}_data.pkl', 'wb'))


# create_per_CC_pixel_data_pickles('ATL', 26, 'iso', 'egfp')
# create_per_CC_pixel_data_pickles('ATL', 26, 'iso', 'mch')

# exit()



def get_region_areas(label_id_junctions):
    """
    Get areas of regions
    @param label_id_junctions: dict of CC ids and junctions
    @return: list of areas
    """
    areas = []
    # for id, juncs in label_id_junctions.items():
    #     if len(juncs) < 500:
    #         areas.append(len(juncs))
    for id, juncs in label_id_junctions.items():
        areas.append(len(juncs))

    return areas


def get_cc_ids(labelled_img, region):
    """
    Get CC ids for the specified region
    @param labelled_img: labelled image
    @param region: isolated or fuzzy
    @return: list of CC ids
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


def get_region_cc(group, series_num, region):
    """
    Get CC ids for the specified region
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param series_num: series number
    @param region: isolated or fuzzy
    @return: list of CC ids
    """
    ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, series_num)

    # dict with ids as key and (x, y) as value
    label_ids = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz = junc_analysis.get_junction_areas(label_ids)

    if region == 'iso':
        return get_cc_ids(labelled_img, iso), labelled_img
    else:
        return get_cc_ids(labelled_img, fuz), labelled_img
    

# def get_num_junctions_per_group(group, num_series):
#     junction_num = []
#     for num in range(1, num_series+1):
#         ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)

#         # dict with ids as key and (x, y) as value
#         label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

#         # iso, fuz, unk: list of lists with x, y
#         iso, fuz, unk = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)

#         junction_num.append(len(iso))
#     return junction_num


def get_region_areas_per_group(group, num_series, region):
    """
    Get areas of regions per group
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param num_series: number of series
    @param region: 'iso' or 'non-iso'
    @return: list of lists of areas
    """
    area_data = []
    for num in range(1, num_series + 1):
        try:
            ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
            label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
            label_id_junctions = get_junctions_per_cc_id(label_ids, per_frame_junctions, labelled_img, region)
            area_vals = get_region_areas(label_id_junctions)
            area_data.append(area_vals)
        except Exception:
            print(f'Series {num} not available')
            continue

    return area_data


def cc_signal(group, channel, region):
    """
    Calculate deposit
    @param num_series: number of series
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'iso' or 'non-iso'
    @return: list of lists of deposits
    """

    groups = {'ATL': ('A', 26), 'Climp': ('C', 31), 'Control': ('Ct', 31), 'RTN': ('R', 29)}
    # groups = {'ATL': 26, 'Climp': 31, 'RTN': 29}
    channel_idx = 0 if channel == 'egfp' else 1

    group_data = []
    for series_num in range(1, groups[group][1] + 1):
        region_cc, labelled_img = get_region_cc(group, series_num, region)

        # CC coords per series
        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        # Data: num_cc * 100
        if region_cc is None or labelled_img is None:
            print(f"Error: Unable to get region connected components for {group} series {series_num}. Skipping.")
            continue

        # CC coords per series
        region_cc_coords = {label: np.where(labelled_img == label) for label in region_cc}

        # Data: num_cc * 100
        series_values = []

        for i in range(100):
            if group == 'Control':
                path = os.path.join(confocal_data_path, 'Control', 'files', f'img_{series_num}_decon_t0{i:02d}.tif')
            else:
                path = os.path.join(confocal_data_path, group, 'files', f'{groups.get(group, ("", ""))[0]}{series_num}_decon_t0{i:02d}_ch0{channel_idx}.tif')

            # if not os.path.exists(path):
            #     print(f"Error: File not found - {path}. Skipping.")
            #     continue

            img = get_std_img(path)

            # if img is None:
            #     print(f"Error: Unable to read image from {path}. Skipping.")
            #     continue

            region_means = [np.mean(img[coords]) for coords in region_cc_coords.values()]
            series_values.append(region_means)

        if series_values:
            series_values = np.array(series_values)
            group_data.append(series_values.T)

    return group_data        


# calc_deposit_net_norm
def cc_signal_net_norm(num_series, group, region):
    """
    Calculate deposit
    @param num_series: number of series
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'iso' or 'fuz'
    @return: list of lists of deposits
    """

    # Get CC ids for the specified region
    region_cc, labelled_img = get_region_cc(group, num, region)
    region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

    egfp_seq_data = []
    mch_seq_data = []

    # mnmx_egfp = []
    # mnmx_mch = []

    if group == 'Control':
        path_skel = f'{confocal_data_path}/{group}/new_op_jul/skel_max_proj/Ct{num}_max.png'
    else:
        path_skel = f'{confocal_data_path}/{group}/new_op_jul/skel_max_proj/{group[0]}{num}_max.png'

    for i in range(100):
        if group == 'Control':
            path_EGFP = f'{confocal_data_path}/Control/files/img_{num}_decon_t0{i:02d}.tif'
            path_mch = None
        else:
            path_EGFP = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'
            path_mch = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch01.tif'
            # path_skel = f'{confocal_data_path}/{group}/new_op_jul/skel/{group[0]}{num}/{group[0]}{num}_decon_t0{i:02d}_ch00_skel.png'

        # Read images and get extract data
        img_egfp = imageio.imread(path_EGFP)
        skel = imageio.imread(path_skel)
        data_egfp = img_egfp[np.where(skel)]
        egfp_norm = (img_egfp - min(data_egfp)) / (max(data_egfp) - min(data_egfp))
        egfp_frame_data = [np.mean(egfp_norm[v]) for v in region_cc_coords.values()]

        egfp_seq_data.append(egfp_frame_data)

        if path_mch is not None:
            img_mch = imageio.imread(path_mch)

            data_mch = img_mch[np.where(skel)]

            mch_norm = (img_mch - min(data_mch)) / (max(data_mch) - min(data_mch))

            mch_frame_data = [np.mean(mch_norm[v]) for v in region_cc_coords.values()]

            mch_seq_data.append(mch_frame_data)

    egfp_seq_data = np.array(egfp_seq_data)

    mch_seq_data = np.array(mch_seq_data)

    return egfp_seq_data.T, mch_seq_data.T, region_cc_coords


def cc_signal_cc_norm(num_series, group, region):
    """
    Calculate deposit
    @param num_series: number of series
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'iso' or 'non-iso'
    @return: list of lists of deposits
    """


    region_cc, labelled_img = get_region_cc(group, num, region)

    region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

    egfp_seq_data = []
    mch_seq_data = []

    for i in range(100):
        if group == 'Control':
            path_EGFP = f'{confocal_data_path}/Control/files/img_{num}_decon_t0{i:02d}.tif'
            path_mch = None
        else:
            path_EGFP = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'
            path_mch = f'{confocal_data_path}/{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch01.tif'

        img_egfp = imageio.imread(path_EGFP)

        # img_egfp = get_std_img(path_EGFP)
        egfp_frame_data = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]

        egfp_seq_data.append(egfp_frame_data)
        # print(region_cc_coords.values())
        # exit()
        # frame_data = []

        # egfp_frame_data = [np.mean(img_egfp[v]) for v in region_cc_coords.values()]
        # for v in region_cc_coords.values():
        #     std_junc = (img_egfp[v] - min(img_egfp[v])) / (max(img_egfp[v]) - min(img_egfp[v]))
        # frame_data.append(img_egfp[v])
        # print(std_junc)
        # exit()
        # frame_data.append(np.mean(std_junc))

        # print(frame_data)
        # exit()

        # frame_data = (frame_data - min(frame_data)) / (max(frame_data) - min(frame_data))

        # print(frame_data)
        # exit()

        # egfp_seq_data.append(egfp_frame_data)
        # egfp_seq_data.append(frame_data)

        if path_mch is not None:
            img_mch = imageio.imread(path_mch)

            mch_frame_data = [np.mean(img_mch[v]) for v in region_cc_coords.values()]

            mch_seq_data.append(mch_frame_data)

    egfp_seq_data = np.array(egfp_seq_data)

    mch_seq_data = np.array(mch_seq_data)

    op_egfp = []
    op_mch = []

    for each in egfp_seq_data.T:
        each = (each - each.min()) / (each.max() - each.min())
        op_egfp.append(each)

    for each in mch_seq_data.T:
        each = (each - each.min()) / (each.max() - each.min())
        op_mch.append(each)

    op_egfp = np.array(op_egfp)
    op_mch = np.array(op_mch)

    # return egfp_seq_data.T, mch_seq_data.T, region_cc_coords
    return op_egfp, op_mch, region_cc_coords


def calc_egfp_deposit(group, channel, num_series, region):
    """
    Calculate EGFP deposit
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param channel: 'EGFP' or 'mCherry'
    @param num_series: number of series
    @param region: 'iso' or 'non-iso'
    @return: list of lists of deposits
    """

    channel_idx = 1 if channel == 'mCherry' else 0
    series_data = []

    for series_num in range(1, num_series + 1):
        region_cc, labelled_img = get_region_cc(group, series_num, region)
        region_cc_coords = {each: np.where(labelled_img == each) for each in region_cc}

        series_values = []
        for i in range(100):
            if group == 'Control':
                path = f'{confocal_data_path}/Control/files/img_{series_num}_decon_t0{i:02d}.tif'
            else:
                path = f'{confocal_data_path}/{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch0{channel_idx}.tif'
            img = get_std_img(path)
            region_means = [np.mean(img[coords]) for coords in region_cc_coords.values()]
            series_values.extend(region_means)

        series_data.extend(np.array(series_values))

    return series_data


def cc_area_measure(group, region, rstart, rend):
    """
    Calculate the area of each connected component
    @param group: 'ATL', 'Climp', 'Control', 'RTN'
    @param region: 'isolated' or 'fuzzy'
    @param rstart: start series number
    @param rend: end series number
    @return: list of areas
    """
    cc_area_list = []

    for series_num in range(rstart, rend + 1):
        try:
            region_cc, labelled_img = get_region_cc(group, series_num, region)
            regions = regionprops(labelled_img)

            if len(regions) == 0:
                print("No connected components in series %d" % series_num)
                continue

            cc_areas = [regions[each - 1]['Area'] for each in region_cc]
            cc_area_list.extend(cc_areas)
        except Exception:
            print(f'Series {series_num} not available')
            continue

    # with open(f'cc_area_{group}_{region}.pkl', 'wb') as f:
    #     pickle.dump(cc_area_list, f)

    return cc_area_list


def stat_analysis(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl):
    """
    Perform statistical analysis
    @param cc_area_atl: list of areas of ATL
    @param cc_area_climp: list of areas of Climp
    @param cc_area_rtn: list of areas of RTN
    @param cc_area_ctrl: list of areas of Control
    @return: None
    """
    f_stat, p_val = f_oneway(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)

    mc = MultiComparison(pd.concat([cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl]), pd.Series(
        ["ATL"] * len(cc_area_atl) + ["Climp"] * len(cc_area_climp) + ["RTN"] * len(cc_area_rtn) + ["Control"] * len(
            cc_area_ctrl)))
    result = mc.tukeyhsd()

    stat, p_val = kruskal(cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)

    series_pairs = [(cc_area_atl, cc_area_climp), (cc_area_atl, cc_area_rtn), (cc_area_atl, cc_area_ctrl),
                    (cc_area_climp, cc_area_rtn), (cc_area_climp, cc_area_ctrl), (cc_area_rtn, cc_area_ctrl)]

    for pair in series_pairs:
        u_stat, p_val = mannwhitneyu(pair[0], pair[1], alternative='two-sided')
        print("Mann-Whitney U test between", pair[0].name, "and", pair[1].name)
        print("U-statistic:", u_stat)
        print("p-value:", p_val)



def junc_area_locator(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: dt, dictionary with matched junctions per reference junction - nearest neighbour approach
    """
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    er_img = ''
    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

    # Get junction coordinates from projection frame
    # newps = junc_analysis.get_junctions(er_img, mean_img)
    graph = junc_analysis.skel_to_graph(mean_img)
    newps = junc_analysis.get_ref_junctions(graph)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)
    # print(nps.shape)

    # sort the array for nearest neighbour matching per frame
    # nps_sorted is the mean projection (reference) junction list
    nps_sorted = sorted(nps, key=lambda t: t[0])

    dt = {}
    for frame in range(100):

        er_img = ''
        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

        sk_newps = junc_analysis.get_junctions(er_img, sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        sk_nnode_coords = np.array(sk_nps)

        # sk_nps_sorted is the per frame junction list
        sk_nps_sorted = sorted(sk_nps, key=lambda t: t[0])

        # matching of junction candidates
        for elem in nps_sorted:
            for sk_elem in sk_nps_sorted:
                dst = np.sqrt((sk_elem[0] - elem[0]) ** 2 + (sk_elem[1] - elem[1]) ** 2)
                if dst < 3:
                    if (elem[0], elem[1]) not in dt.keys():
                        dt[(elem[0], elem[1])] = []
                    dt[(elem[0], elem[1])].append([sk_elem, frame, dst])
    return dt


def refine_junc_dt(dt, min_presence=50):
    return {k: v for k, v in dt.items() if len(v) > min_presence}



def get_label_id(regions, iso, junc_id):
    for j in range(len(regions)):
        region_coords = regions[j].coords
        for each in region_coords:
            a, b = each[0], each[1]
            if a == iso[junc_id][0] and b == iso[junc_id][1]:
                return j


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure





def per_frame_num_junctions(labelled_img, iso):
    iso_cc = get_cc_ids(labelled_img, iso)

    # get lists to store the count of junctions within CC per frame
    iso_junc_num = []
    fuz_junc_num = []

    for i in range(100):
        temp_iso = []
        temp_fuz = []
        junc_frame = imageio.imread(
            confocal_data_path + f'ATL/new_op_jul/junctions/A1/A1_decon_t0{i:02d}_ch00_junc.png')

        # get the junc locations
        locations = np.where(junc_frame > 0)

        for locx, locy in zip(locations[0], locations[1]):
            cc_id = labelled_img[locx, locy]
            if cc_id in iso_cc:
                temp_iso.append(cc_id)
            else:
                temp_fuz.append(cc_id)
        iso_junc_num.append(temp_iso)
        fuz_junc_num.append(temp_fuz)

    return iso_junc_num, fuz_junc_num




def junction_location_plotter(group, num_series):
    # mean_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
    mean_img = confocal_data_path + f'{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

    graph = junc_analysis.skel_to_graph(mean_img)
    newps = junc_analysis.get_ref_junctions(graph)

    nps = [[each[0], each[1]] for each in newps]
    nps = np.array(nps)

    # print(nps[46, 1], nps[46, 0])
    #
    # exit()
    # plt.imshow(mean_proj_img, cmap='gray')

    # for (s,e) in g.edges():
    #     ps = g[s][e]['pts']
    #     plt.plot(ps[:,1], ps[:,0], 'green')
    # for dr in range(len(nps)):
    #     os.makedirs('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A1_junc_viz/junc%s'%f'{dr+1}')

    for frame in range(100):

        # ER Input image
        if group == 'Control':
            path = confocal_data_path + f'{group}/files/img_{num_series}_decon_t0{frame:02d}.tif'

        else:
            path = confocal_data_path + f'{group}/files/{group[0]}{num_series}_decon_t0{frame:02d}_ch00.tif'

        img = imageio.imread(path)
        img = (img - img.min()) / (img.max() - img.min())

        # ER Skel image
        # sk_img = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0%s_ch00_skel.png'%f'{i:02d}'
        sk_img = confocal_data_path + f'{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        sk_nps = np.array(sk_nps)

        # img = img * 255.
        # # plt.plot(ps[:, 1], ps[:, 0], 'y.')
        # for j in range(47, 48):
        # plt.figure(figsize=(128/77, 128/77))
        plt.imshow(img, cmap='gray')
        # plt.imshow(imageio.imread(img), cmap='gray')
        plt.axis('off')
        # plt.title('t=%s'%f'{i}')
        plt.plot(nps[:, 1], nps[:, 0], 'r.')
        plt.plot(sk_nps[:, 1], sk_nps[:, 0], 'b.')
        # y = nps[0, 1]
        # x = nps[0, 0]
        # cv2.rectangle(img, (x-1, y-1), (x+1, y+1), (0, 0, 255), 2)

        plt.savefig((
                                confocal_data_path + f'{group}/new_op_jul/{group.lower()}_junc_viz/{group_pref[group]}{num_series}_t{frame:02d}.png'),
                    bbox_inches='tight', pad_inches=0)

        plt.close()


# junction_location_plotter('RTN', 1)
# exit()
