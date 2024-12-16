"""
Extracting and flatmapping projection locations
Codebase by Michael Reimann, adaptation to this use-case Christoph Pokorny and András Ecker
Last updates: 02/2022 (A. Ecker)
              09/2023 (C. Pokorny; converted to SONATA configs)
"""

import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from voxcell import VoxcellError
from voxcell.nexus.voxelbrain import Atlas
from flatmap_utility import per_pixel_coordinate_transformation

CELLS_XYZ = ["x", "y", "z"]
VIRTUAL_FIBERS_XYZ = ["x", "y", "z"]
VIRTUAL_FIBERS_UVW = ["u", "v", "w"]
FIX_TRANSITION = 1500  # In the SSCx Bio_M the projection fiber positions are defined at the start of the rays,
                       # which are outside of Sirio's flatmap. They have to be shifted with 1500 (as of 11.2021)
                       # to get them into the flatmap.


# def _projection_locations_3d(projection):
#     """Reads projection locations from virtual nodes population"""
#     assert projection.type == "virtual", "ERROR: Projection must be a 'virtual' nodes population!"
#     fib_props = projection.get()  # Properties table of projecting fibers
#     return fib_props.index.values, fib_props[VIRTUAL_FIBERS_XYZ].values, fib_props[VIRTUAL_FIBERS_UVW].values


def _neuron_locations_3d(nodes, target_name):
    """Returns 3D locations of neurons from nodes population"""
    pos = nodes.positions(group=target_name)
    return pos.index.values, pos[CELLS_XYZ].values


def apply_flatmap(xyz, uvw, fm, max_translation=2000):
    """
    Uses Voxcell to look up locations of `xyz` in the flatmap. If locations are outside the valid region of
    the flatmap (which is usually the case for projections as those start at the bottom of L6)
    and `uvw` is provided (i.e. not `None`), then there are 2 possible options:
    1) The invalid locations are translated along the directions given by `uvw` by a hard-coded fix amount, or
    2) The invalid locations are gradually translated along the directions given by `uvw` until they hit
    the valid volume. `max_translation` defines the maximum amplitude of that translation.
    Locations that never hit the valid volume will return a flat location of (-1, -1).
    :param xyz: numpy.array, N x 3: coordinates in 3d space
    :param uvw: numpy.array, N x 3: directions in 3d space. Optional, can be None.
    :param fm: VoxelData: flatmap
    :param max_translation: float.
    :return: Flat locations of the xyz coordinates in the flatmap.
    """
    solution = fm.lookup(xyz)
    if uvw is not None:
        # 1)
        solution = fm.lookup(xyz + FIX_TRANSITION * uvw)
        if np.all(solution > 0):
            return solution
        else:
            # 2)
            fac = 0
            step = fm.voxel_dimensions[0] / 4
            tl_factors = np.zeros((len(uvw), 1))
            solution = fm.lookup(xyz)
            while np.any(solution < 0) and fac < max_translation:
                try:
                    fac += step
                    to_update = np.any(solution < 0, axis=1)
                    tl_factors[to_update, 0] = fac
                    solution[to_update, :] = fm.lookup(xyz[to_update, :] + tl_factors[to_update, :] * uvw[to_update, :])
                except VoxcellError:
                    break
    return solution

def _get_atlas(c):
    """Returns atlas extracted from SONATA circuit config components"""
    atlas_dir = c.config.get("components", {}).get("provenance", {}).get("atlas_dir")
    if atlas_dir is not None:
        # print(atlas_dir)
        return Atlas.open(atlas_dir)
    else:
        return None


def get_neuron_locations(c, target_name):
    """Return neuron locations in 2D (if flatmap is available) and 3d (from SONATA nodes population)"""
    # Find SONATA nodes population containing actual neurons (i.e., non-virtual population)
    npop_name = [_pop for _pop in c.nodes.population_names if c.nodes[_pop].type != "virtual"]
    assert len(npop_name) == 1, "ERROR: Single (non-virtual) nodes population expected!"
    nodes = c.nodes[npop_name[0]]

    # Access 2D/3D locations
    nids, pos3d = _neuron_locations_3d(nodes, target_name)
    atlas = _get_atlas(c)
    if atlas is not None:
        fm = atlas.load_data("flatmap")
        pos2d = apply_flatmap(pos3d, None, fm)
        return nids, pos2d, pos3d
    else:
        warnings.warn("No atlas found in SONATA circuit config components, so 2D locations won't be returned")
        return nids, None, pos3d


def mask_results_bb(results, c, mask_name):
    """
    :param results: The unmasked output of `get_projection_locations()`
    :param c: bluepysnap.Circuit
    :param mask_name: str: Name of a cell target of projection that serves as a mask
    :return: The "results" are masked such that only the parts within the bounding box of `mask_name` are returned.
             If the SONATA circuit config has an atlas component and a flatmap can be loaded then the bounding box and masking
             is done in the 2D flat space. Otherwise in 3D space.
    """
    res_nids, res2d, res3d, resdir = results
    _, mask2d, mask3d = get_neuron_locations(c, mask_name)
    if mask2d is None:
        valid = (res3d >= mask3d.min(axis=0, keepdims=True)) & (res3d <= mask3d.max(axis=0, keepdims=True))
    else:
        mask2d = mask2d[np.all(mask2d >= 0, axis=1)]
        valid = (res2d >= mask2d.min(axis=0, keepdims=True)) & (res2d <= mask2d.max(axis=0, keepdims=True))
    valid = np.all(valid, axis=1)

    res_nids = res_nids[valid]
    if res2d is not None:
        res2d = res2d[valid]
    if res3d is not None:
        res3d = res3d[valid]
    if resdir is not None:
        resdir = resdir[valid]

    return res_nids, res2d, res3d, resdir


def mask_results_dist(results, circ, mask_name, max_dist=None, dist_factor=2.0):
    """
    :param results: The unmasked output of `get_projection_locations()`
    :param circ: bluepysnap.Circuit
    :param mask_name: str: Name of a cell target of projection that serves as a mask
    :param max_dist: float: (Optional) Maximal distance from the `mask_name` location that is considered valid.
    If not provided, a value will be estimated using `dist_factor`
    :param dist_factor: float: (Optional, default: 2.0) If `max_dist` is None, this will be used to conduct an estimate.
    :return: The `results` are masked such that only the parts within `max_dist` of locations associated with
             `mask_name` are returned. If the SONATA circuit config has an atlas component and a flatmap can be loaded then
             the bounding box and masking is done in the 2D flat space. Otherwise in 3D space.
    """
    res_nids, res2d, res3d, resdir = results
    _, mask2d, mask3d = get_neuron_locations(circ, mask_name)

    if mask2d is None:
        use_res = res3d
        use_mask = mask3d
    else:
        use_res = res2d
        use_mask = mask2d

    t_res = KDTree(use_res)
    t_mask = KDTree(use_mask)
    if max_dist is None:
        dists, _ = t_res.query(use_res, 2)
        max_dist = dist_factor * dists[:, 1].mean()
    actives = t_mask.query_ball_tree(t_res, max_dist)
    actives = np.unique(np.hstack(actives).astype(int))

    res_nids = res_nids[actives]
    if res2d is not None:
        res2d = res2d[actives]
    if res3d is not None:
        res3d = res3d[actives]
    if resdir is not None:
        resdir = resdir[actives]

    return res_nids, res2d, res3d, resdir


# def get_projection_locations(c, projection_name, mask=None, mask_type="bbox"):
#     """Gets node ids, 2D locations in flat space (if flatmap is available), 3D locations and directions of
#        projections masked with a given target region defined in the circuit."""
#     # get projection locations
#     if projection_name in c.nodes.population_names:
#         nids, pos3d, dir3d = _projection_locations_3d(c.nodes[projection_name])
#     else:
#         raise RuntimeError("Projection: %s is not part of the SONATA circuit config" % projection_name)
#     # apply flatmap
#     atlas = _get_atlas(c)
#     if atlas is not None:
#         fm = atlas.load_data("flatmap")
#         pos2d = apply_flatmap(pos3d, dir3d, fm)
#     else:
#         warnings.warn("No atlas found in SONATA circuit config components, so 2D locations won't be returned.")
#         if mask is not None:
#             warnings.warn("This will seriously affect masking as the 3D locations of the projection fibers are"
#                           "(below L6, thes) outside of the circuit's volume")
#         pos2d = None
#     # mask with region (hopefully) in flat space
#     if mask is not None:
#         if mask_type == "bbox":
#             nids, pos2d, pos3d, dir3d = mask_results_bb((nids, pos2d, pos3d, dir3d), c, mask)
#         elif mask_type.find("dist") == 0:
#             mask_spec = mask_type.replace("dist", "")  # Extract distance factor, e.g. mask_type = "dist2.0"
#             dist_factor = float(mask_spec) if len(mask_spec) > 0 else None
#             nids, pos2d, pos3d, dir3d = mask_results_dist((nids, pos2d, pos3d, dir3d), c, mask, dist_factor)
#         else:
#             raise RuntimeError("Mask type %s unknown!" % mask_type)
#     return nids, pos2d, pos3d, dir3d


def flat_coordinate_frame(pos3d, dir3d, fm, grouped=False):
    """Return same format as flatmap_utility.py/flat_coordinate_frame() but using the local `apply_flatmap()` fn.
    developed for the projections instead of the vanilla `VoxcellData.lookup()`"""
    coords_flat = apply_flatmap(pos3d, dir3d, fm)
    coord_frame = pd.DataFrame(pos3d, columns=["x", "y", "z"],
                               index=(pd.MultiIndex.from_tuples(map(tuple, coords_flat), names=["f_x", "f_y"])))
    if grouped:
        return coord_frame.groupby(["f_x", "f_y"]).apply(lambda x: x.values)
    return coord_frame


def projection_flat_coordinate_frame(nids, pos3d, dir3d, fm, grouped=False):
    """Return same format as flatmap_utility.py/neuron_flat_coordinate_frame() but takes
    precomputed `nids` and `pos3d` as input not a `bluepysnap.Circuit`"""
    coord_frame = flat_coordinate_frame(pos3d, dir3d, fm)
    coord_frame["nid"] = nids
    if grouped:
        A = coord_frame[["x", "y", "z"]].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        B = coord_frame["nid"].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        return A, B
    return coord_frame


def supersampled_projection_locations(nids, pos3d, dir3d, fm, orient, pixel_sz=34.0):
    """Function based on flatmap_utility.py/supersampled_neuron_locations()"""
    proj_loc_frame, proj_nid_frame = projection_flat_coordinate_frame(nids, pos3d, dir3d, fm, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system="subpixel")
    idxx = proj_loc_frame.index.intersection(tf.index)

    res = tf[idxx].combine(proj_loc_frame[idxx], lambda a, b: a.apply(b))
    final = res.index.to_series().combine(res, lambda a, b: np.array(a) * pixel_sz + b)
    final_frame = np.vstack(final.values)
    out = pd.DataFrame(final_frame, columns=["flat x", "flat y"],
                       index=pd.Index(np.hstack(proj_nid_frame[idxx].values), name="nid"))
    return out


def get_projection_locations(c, projection_name, mask=None, mask_type="bbox", supersample=False):
    """Gets node ids, 2D locations in flat space (if flatmap is available), 3D locations and directions of projections
    masked with a given target region defined in the circuit."""

    if projection_name in c.nodes.population_names:
        # nids, pos3d, dir3d = _projection_locations_3d(c.nodes[projection_name])
        converted = get_converted_flatmap()

        # If you're using a circuit which is a subset of the full sscx
        # then the virtual vpm fibers in nodes['VPM'] are only a subset of the full sscx ones
        # This code block finds the corresponding vpm fibers in the full sscx flatmap
        vpm_nodes_df = c.nodes['VPM'].get() 
        print("vpm_nodes_df.columns: " + str(vpm_nodes_df.columns))
        if 's1_id' in vpm_nodes_df.columns:
            print("s1_id")
            s1_ids = vpm_nodes_df['s1_id'] - 5000000
            converted = converted.loc[s1_ids.values].reset_index()

        nids, pos3d, dir3d = _projection_locations_3d(c.nodes['VPM'], converted.index.values)
    else:
        raise RuntimeError("Projection: %s is not part of the SONATA circuit config" % projection_name)
    # apply flatmap
    atlas = _get_atlas(c)
    if atlas is not None:
        fm = atlas.load_data("flatmap")
        # print(fm)
        # print(pos3d)
        # print(dir3d)
        pos2d = apply_flatmap(pos3d, dir3d, fm)
    else:
        warnings.warn("No atlas found in SONATA circuit config components, so 2D locations won't be returned.")
        if mask is not None:
            warnings.warn("This will seriously affect masking as the 3D locations of the projection fibers are"
                          "(below L6, thes) outside of the circuit's volume")
        pos2d = None
    # mask with region (hopefully) in flat space
    if mask is not None:
        if mask_type == "bbox":
            nids, pos2d, pos3d, dir3d = mask_results_bb((nids, pos2d, pos3d, dir3d), c, mask)
        elif mask_type.find("dist") == 0:
            mask_spec = mask_type.replace("dist", "")  # Extract distance factor, e.g. mask_type = "dist2.0"
            dist_factor = float(mask_spec) if len(mask_spec) > 0 else None
            nids, pos2d, pos3d, dir3d = mask_results_dist((nids, pos2d, pos3d, dir3d), c, mask, dist_factor)
        else:
            raise RuntimeError("Mask type %s unknown!" % mask_type)
    # supersample (only the masked parts)
    if supersample:
        if atlas:
            fm = atlas.load_data("flatmap")
            orient = atlas.load_data("orientation")
        else:
            raise RuntimeError("Please add Atlas to the SONATA circuit config as it's used to load the flatmap and orientation!")
        super_pos_frame = supersampled_projection_locations(nids, pos3d, dir3d, fm, orient)
        idx = np.argsort(super_pos_frame.index.to_numpy())
        pos2d = super_pos_frame.values[idx]

    # return None, None, None, None
    return nids, pos2d, pos3d, dir3d


def mask_location_by_dist(nids, pos, max_dist):
    """Unlike `mask_results_bb()` and `mask_results_dist()` above this one applies and extra mask on fiber locations
    based on `max_dist` before clustering (just to avoid boundary artifacts)"""
    center = np.mean(pos, axis=0)
    dists = np.sqrt(np.sum((pos - center)**2, axis=1))
    idx = np.where(dists < max_dist)[0]
    return nids[idx], pos[idx]


def cluster_by_locations(nids, pos, n_clusters=None, n_per_cluster=None, cluster_seed=0):
    """
    :param nids: numpy.array, N x 1: List virtual projection node IDs
    :param pos: numpy.array, N x D: D-dim locations of the projection fibers (2d or 3d)
    :param n_clusters: int: Number if clusters. Optional, can be None if n_per_cluster is given.
    :param n_per_cluster: int: Number of fibers per cluster. Optional, can be None if n_clusters is given.
    :param cluster_seed: int: Random seed of k-means clustering. Optional, default: 0.
    :return:
    The list of lists of nids belonging to a cluster of nearby fibers, D-dim cluster centroids, and cluster
    indices associated with the clusters of projection fibers.
    Either "n_clusters" or "n_per_cluster" needs to be specified to determine the resulting number of clusters.
    "pos" can be either 2d or 3d positions.
    """
    if n_clusters is None:
        if n_per_cluster is None:
            raise RuntimeError("Need to specify number of clusters or mean number of fibers per cluster")
        n_clusters = int(round(float(len(nids)) / n_per_cluster))

    if n_clusters == len(nids): # No clustering (i.e., 1 fiber = 1 cluster)
        nids_list = [[i] for i in range(len(nids))]
        cluster_pos = pos
        cluster_idx = np.arange(len(nids))
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=cluster_seed).fit(pos)
        nids_list = [nids[kmeans.labels_ == i] for i in range(n_clusters)]
        cluster_pos = kmeans.cluster_centers_
        cluster_idx = kmeans.labels_

    return nids_list, cluster_pos, cluster_idx


def plot_clusters_of_fibers(grp_idx, grp_pos, pos2d, pos3d, pos2d_all, pos3d_all, save_path=None):
    """
    Plots spatial clusters (groups) of nearby fibers
    """
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    num_groups = grp_pos.shape[0]
    grp_colors = plt.cm.jet(np.linspace(0, 1, num_groups))
    np.random.seed(0) # Just for color permutation
    grp_colors = grp_colors[np.random.permutation(num_groups), :]

    if pos2d is not None:
        plt.figure(figsize=(5, 5))
        plt.plot(pos2d_all[:, 0], pos2d_all[:, 1], '.', color='grey', markersize=1)
        for i in range(num_groups):
            plt.plot(pos2d[grp_idx == i, 0], pos2d[grp_idx == i, 1], '.', color=grp_colors[i, :], markersize=1)
        if grp_pos.shape[1] == 2:
            for i in range(num_groups):
                plt.plot(grp_pos[i, 0], grp_pos[i, 1], 'x', color=grp_colors[i, :])
        plt.axis('image')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Clusters of fibers')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'clusters_2d.png'), dpi=300)

    if pos3d is not None:
        plt.figure(figsize=(5, 5))
        plt.subplot(1, 1, 1, projection='3d')
        plt.plot(pos3d_all[:, 0], pos3d_all[:, 1], pos3d_all[:, 2], '.', color='grey', markersize=1)
        for i in range(num_groups):
            plt.plot(pos3d[grp_idx == i, 0], pos3d[grp_idx == i, 1], pos3d[grp_idx == i, 2], '.', color=grp_colors[i, :], markersize=1)
        if grp_pos.shape[1] == 3:
            for i in range(num_groups):
                plt.plot(grp_pos[i, 0], grp_pos[i, 1], grp_pos[i, 2], 'x', color=grp_colors[i, :])
        plt.gca().set_xlabel('x')
        plt.gca().set_ylabel('y')
        plt.gca().set_zlabel('z')
        plt.title('Clusters of fibers')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, 'clusters_3d.png'), dpi=300)


def plot_cluster_size_distribution(grp_idx, save_path=None):
    """
    Plot distribution of cluster (group) sizes
    """
    num_clusters = np.max(grp_idx) + 1

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    cl_hist = plt.hist(grp_idx, bins=np.arange(-0.5, num_clusters + 0.5, 1))[0]
    plt.text(0.99 * np.max(plt.xlim()), 0.99 * np.max(plt.ylim()), f'MIN: {np.min(cl_hist)}\nMAX: {np.max(cl_hist)}\nMEAN: {np.mean(cl_hist):.1f}\nSTD: {np.std(cl_hist):.1f}\nCOV: {np.std(cl_hist) / np.mean(cl_hist):.1f}', ha='right', va='top')
    plt.xlim(plt.xlim()) # Freeze axis limits
    plt.plot(plt.xlim(), np.full(2, np.mean(cl_hist)), '--', color='tab:red')
    plt.xlabel('Cluster idx')
    plt.ylabel('Cluster size')
    plt.title(f'Cluster sizes (N={num_clusters})')

    plt.subplot(1, 2, 2)
    plt.hist(cl_hist, bins=np.arange(-0.5, np.max(cl_hist) + 1.5))
    plt.ylim(plt.ylim()) # Freeze axis limits
    plt.plot(np.full(2, np.mean(cl_hist)), plt.ylim(), '--', color='tab:red')
    plt.xlabel('Cluster size')
    plt.ylabel('Count')
    plt.title('Cluster size distribution')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'cluster_sizes.png'), dpi=300)




"""
JONI FIX
"""
import numpy as np
import pandas as pd
from bluepysnap import Circuit
from voxcell.nexus.voxelbrain import Atlas
NPIX = 190
FIX_TRANSITION = 1500
SIRIO_FLATMAP = '/gpfs/bbp.cscs.ch/data/scratch/proj83/home/bolanos/circuits/Bio_M/20200805/virtual_fibers_gen/original_run/allfibers_VPM_gid_fx_fy.tsv'
# CIRCUIT = Circuit("/gpfs/bbp.cscs.ch/project/proj83/home/isbister/overnight_circuit_config.json")
# FLATMAP = Atlas.open(CIRCUIT.config.get("components", {}).get(
#     "provenance", {}).get("atlas_dir")).load_data("flatmap")


def _projection_locations_3d(projection, ids):
    """Reads projection locations from virtual nodes population"""
    print(ids)
    assert projection.type == "virtual", "ERROR: Projection must be a 'virtual' nodes population!"
    fib_props = projection.get(ids)  # Properties table of projecting fibers
    return fib_props.index.values, fib_props[list('xyz')].values, fib_props[list('uvw')].values


def _convert_xy(flatmap):
    xy = np.uint16(np.floor(flatmap[list('xy')] * (NPIX - 1E-9)))
    xy[:, 1] = NPIX - xy[:, 1]  # it seems y-axis is flipped
    return xy


def get_converted_flatmap():
    df = pd.read_csv(SIRIO_FLATMAP, sep='\t', header=None, names=['sgid', 'x', 'y'])
    df = df.set_index('sgid', drop=True)
    df.index -= 5000000  # remove the offset
    df[list('xy')] = _convert_xy(df)
    return df


# def get_flat_pos_atlas(xyz, uvw):
#     return FLATMAP.lookup(xyz + FIX_TRANSITION * uvw, outer_value=np.array([-1, -1]))


# def compare(gid_pos_dir, converted):
#     gid, pos, dir_ = gid_pos_dir

#     real_positions = get_flat_pos_atlas(pos, dir_)
#     skipmask = np.all(real_positions != -1, axis=1)

#     real_positions = real_positions[skipmask]
#     conv_positions = converted.values[skipmask]
#     diff = real_positions-conv_positions
#     print('max:', np.max(diff, axis=0))
#     print('mean:', np.mean(diff, axis=0))
#     print('mean (abs):', np.mean(abs(diff), axis=0))




