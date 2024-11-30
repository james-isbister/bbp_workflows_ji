"""
Sets up random patterns (groups of VPM gids) with some overlap             J
The idea behind the pyramid like overlap comes from topology:             H I
as we want to compare the distance of the patterns                       E F G
to the distance of the (connectivity graphs of the) plastic circuits.   A B C D
last updates: András Ecker, 11.2021
              Christoph Pokorny, 09.2023 (added base pattern generator w/o overlap): A B C D E F G H
"""

import os
import pickle
import numpy as np
from functools import reduce

FIRST_ROW = ["A", "B", "C", "D"]
BASE_PATTERNS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _select_percent(array, percent, seed):
    """Randomly selects a given percent of the input array"""
    n_sample = int(len(array)*percent)
    np.random.seed(seed)
    return np.random.choice(array, n_sample, replace=False)


def _get_second_row(patterns, first_row):
    """Gets 2nd row of the pyramid by combining one half of the first row ones"""
    # get half of each of the first row ones
    tmp = {name: _select_percent(patterns[name], 0.5, 12+seed) for seed, name in enumerate(first_row)}
    # combine them to get the second row of the pyramid
    for first_half, second_half, name in zip(first_row[:-1], first_row[1:], ["E", "F", "G"]):
        patterns[name] = np.union1d(tmp[first_half], tmp[second_half])
    return patterns


def _get_third_row(patterns, first_row):
    """Gets 3rd row of the pyramid by combining one third of the first row ones"""
    # get one third of each of the base ones
    tmp = {name: _select_percent(patterns[name], 1/3., 123+seed) for seed, name in enumerate(first_row)}
    # combine them to get the third row of the pyramid
    patterns["H"] = reduce(np.union1d, (tmp["A"], tmp["B"], tmp["C"]))
    patterns["I"] = reduce(np.union1d, (tmp["B"], tmp["C"], tmp["D"]))
    return patterns


def _get_fourth_row(patterns, first_row):
    """Gets 4th row of the pyramid by combining one fourth of the first row ones"""
    # get one fourth of each of the base ones
    tmp = {name: _select_percent(patterns[name], 0.25, 1234+seed) for seed, name in enumerate(first_row)}
    # combine all to get the fourth row of the pyramid
    patterns["J"] = reduce(np.union1d, (tmp["A"], tmp["B"], tmp["C"], tmp["D"]))
    return patterns


def extend_patterns(patterns, first_row):
    """Builds 6 more patterns in a pyramid like arrangement on top of the first 4 base patterns"""
    patterns = _get_second_row(patterns, first_row)
    patterns = _get_third_row(patterns, first_row)
    patterns = _get_fourth_row(patterns, first_row)
    return patterns


def setup_pyramid_patterns(gids, grp_idx, pct, save_name=None, seed=12345):
    """
    Sets up 4 base patterns and 6 more by combining these (ordered as a pyramid)
    To deal with clusters not having the same size in k-means first preselects clusters of gids
    (close to the 75th percentile of the cluster size distribution) and delete gids from those
    in order to match the min of the selected ones (this way the selected ones will have exactly the same number of gids
    and that number should be close to the mean of the original cluster size).
    :param gids: projection gids (used only at the end for mapping, random selection is based on the grouped version)
    :params grp_idx: idx of grouped gids returned by `lookup_projection_locations.py/cluster_by_locations()`
    :param pct: percentage (0-100) of gids (actually grouped gids...) to use for 1 pattern
    :param save_name: pickle file name to save `proj_patterns`
    :param seed: random seed for the selection of the initial 4 patterns
                 (the seeds of the overlaps/combinations are hard coded in the functions above...)
    :return: proj_patterns: dict with projection gids belonging to given patterns
    """
    # preselect groups of gids around the 75th percentile of the cluster size distribution (see docstring above)
    unique_grp_idx, len_grps = np.unique(grp_idx, return_counts=True)
    n_sample = int(len(unique_grp_idx) * pct / 100)
    first_row_n_sample = len(FIRST_ROW) * n_sample
    len_diffs = np.abs(len_grps - np.percentile(len_grps, 75))
    viable_grp_idx = np.argpartition(len_diffs, first_row_n_sample)[:first_row_n_sample]
    # "delete" random gids from the selected groups to make them exactly the same size (see docstring above)
    len_diffs = len_grps[viable_grp_idx] - len_grps[viable_grp_idx].min()
    for grp_id, diff in zip(viable_grp_idx, len_diffs):
        if diff > 0:
            np.random.seed(seed)
            grp_idx[np.random.choice(np.where(grp_idx == grp_id)[0], diff, replace=False)] = -1
    # get 4 non-overlapping sets as first row of the pyramid
    patterns = {}
    for name in FIRST_ROW:
        tmp = np.arange(len(viable_grp_idx))
        np.random.seed(seed)
        idx = np.random.choice(tmp, n_sample, replace=False)
        patterns[name] = np.sort(viable_grp_idx[idx])
        viable_grp_idx = np.delete(viable_grp_idx, idx)
    # get 6 more by various overlaps of the base ones
    patterns = extend_patterns(patterns, FIRST_ROW)
    fix_patterns(patterns)  # getting rid of rounding errors (if any)
    # get actual gids od projections not just the idx of groups of gids
    pattern_gids = {name: np.sort(np.concatenate([gids[grp_idx == grp_id] for grp_id in idx]))
                    for name, idx in patterns.items()}
    if save_name is not None:
        if not os.path.isdir(os.path.dirname(save_name)):
            os.mkdir(os.path.dirname(save_name))
        with open(save_name, "wb") as f:
            pickle.dump(pattern_gids, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pattern_gids


def setup_base_patterns(gids, grp_idx, pct, save_name=None, seed=12345):
    """
    Sets up 8 base patterns w/o any overlap
    To deal with clusters not having the same size in k-means first preselects clusters of gids
    (close to the 75th percentile of the cluster size distribution) and delete gids from those
    in order to match the min of the selected ones (this way the selected ones will have exactly the same number of gids
    and that number should be close to the mean of the original cluster size).
    :param gids: projection gids (used only at the end for mapping, random selection is based on the grouped version)
    :params grp_idx: idx of grouped gids returned by `lookup_projection_locations.py/cluster_by_locations()`
    :param pct: percentage (0-100) of gids (actually grouped gids...) to use for 1 pattern
    :param save_name: pickle file name to save `proj_patterns`
    :param seed: random seed for the selection of the initial 4 patterns
                 (the seeds of the overlaps/combinations are hard coded in the functions above...)
    :return: proj_patterns: dict with projection gids belonging to given patterns
    """
    # preselect groups of gids around the 75th percentile of the cluster size distribution (see docstring above)
    unique_grp_idx, len_grps = np.unique(grp_idx, return_counts=True)
    n_sample = int(len(unique_grp_idx) * pct / 100)
    base_row_n_sample = len(BASE_PATTERNS) * n_sample
    len_diffs = np.abs(len_grps - np.percentile(len_grps, 75))
    viable_grp_idx = np.argpartition(len_diffs, base_row_n_sample)[:base_row_n_sample]
    # "delete" random gids from the selected groups to make them exactly the same size (see docstring above)
    len_diffs = len_grps[viable_grp_idx] - len_grps[viable_grp_idx].min()
    for grp_id, diff in zip(viable_grp_idx, len_diffs):
        if diff > 0:
            np.random.seed(seed)
            grp_idx[np.random.choice(np.where(grp_idx == grp_id)[0], diff, replace=False)] = -1
    # get 8 non-overlapping sets as base (first row) of the pyramid
    patterns = {}
    for name in BASE_PATTERNS:
        tmp = np.arange(len(viable_grp_idx))
        np.random.seed(seed)
        idx = np.random.choice(tmp, n_sample, replace=False)
        patterns[name] = np.sort(viable_grp_idx[idx])
        viable_grp_idx = np.delete(viable_grp_idx, idx)
    # get actual gids od projections not just the idx of groups of gids
    pattern_gids = {name: np.sort(np.concatenate([gids[grp_idx == grp_id] for grp_id in idx]))
                    for name, idx in patterns.items()}
    if save_name is not None:
        if not os.path.isdir(os.path.dirname(save_name)):
            os.mkdir(os.path.dirname(save_name))
        with open(save_name, "wb") as f:
            pickle.dump(pattern_gids, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pattern_gids


def _update_patterns(patterns, target_pattern, base_pattern):
    """Update patterns by adding an id from `base_pattern` to `target_pattern`"""
    for id_ in patterns[base_pattern]:
        if id_ not in patterns[target_pattern]:
            patterns[target_pattern] = np.concatenate((patterns[target_pattern], np.array([id_])))
            break


def fix_patterns(patterns):
    """Fix number of idx in the upper rows of the pyramid (getting rid of rounding errors)"""
    target_n = len(patterns["A"])
    for base, target in zip(["A", "B", "C"], ["E", "F", "G"]):
        if len(patterns[target]) != target_n:
            _update_patterns(patterns, target, base)
    for base in ["A", "B"]:  # C
        if len(patterns["H"]) != target_n:
            _update_patterns(patterns, "H", base)
    for base in ["B", "C"]:  # D
        if len(patterns["I"]) != target_n:
            _update_patterns(patterns, "I", base)
    for base in ["A", "B", "C"]:  # D
        if len(patterns["J"]) != target_n:
            _update_patterns(patterns, "J", base)



