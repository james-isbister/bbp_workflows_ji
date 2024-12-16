# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation
#                campaigns for pure SONATA circuits
# Author:        C. Pokorny (modified from Pokorny/Reimann/Ecker's earlier code bases)
# Date:          30/08/2023
# Last modified: 12/09/2023

import hashlib
import json
import os
import pickle
import numpy as np
from copy import deepcopy
from bluepysnap import Circuit
from scipy.spatial import distance_matrix
import lookup_projection_locations_SONATA as projloc
import stimulus_generation as stgen
import spikewriter


def _get_cfg_hash(cfg):
    """
    Generates MD5 hash code for given config dict
    """
    hash_obj = hashlib.md5()

    def sort_dict(data):
        """
        Sort dict entries recursively, so that the hash function
        does not depend on the order of elements
        """
        if isinstance(data, dict):
            return {key: sort_dict(data[key]) for key in sorted(data.keys())}
        else:
            return data

    # Generate hash code from sequence of cfg params (keys & values)
    for k, v in sort_dict(cfg).items():
        hash_obj.update(str(k).encode('UTF-8'))
        hash_obj.update(str(v).encode('UTF-8'))
    
    return hash_obj.hexdigest()


def generate_random_dot_flash_stimulus(*, path, sim_duration, **kwargs):
    """
    Generates 'random dot flash' type of stimulus file and writing the
    spikes file(s) to hashed folders to prevent multiple generation of
    exact same spike files
    """

    # _Init_

    ## Get stim config parameters and hash code
    param_list = ['circuit_config', # Circuit
                  'proj_name', 'proj_mask', 'num_fibers_per_cluster', 'n_clusters', 'stimuli_seeds', 'stimuli_seeds_nopt', 'sparsity', # Spatial params
                  'num_stimuli', 'series_seed', 'strict_enforce_p', 'p_seeds', 'overexpressed_tuples', # Series params
                  'start', 'duration_stim', 'duration_blank', 'rate_min', 'rate_max', # Rate signal params
                  'spike_seed', 'bq', 'tau'] # Spike params
    # NOTE: 'num_fibers_per_cluster' and 'n_clusters' mutually exclusive!!

    cfg = {'stim_name': 'RandomDotFlash'}
    cfg.update({p: kwargs.get(p) for p in param_list}) # Stim config parameters
    cfg_hash = _get_cfg_hash(cfg)

    ## Define paths and files
    spikes_path = os.path.join(os.path.split(path)[0], 'spikes', cfg_hash)
    figs_path = os.path.join(spikes_path, 'figs')
    stim_file = os.path.join(spikes_path, 'input.dat')
    rel_stim_file = os.path.relpath(stim_file, path) # Relative path to current simulation folder
    props_file = os.path.splitext(stim_file)[0] + '.json'

    ## Check if stimulus folder for given parameter configuration already exists (using hash code as folder name)
    ## [IMPORTANT: Thread-safe implementation w/o os.path.exists followed by os.makedirs, since this would create
    ##             a race condition in case max_workers > 1 parallel processes are used!]
    try:
        os.makedirs(spikes_path)
        os.makedirs(figs_path)
    except FileExistsError: # Stimulus folder already generated, stop here!
        print(f'INFO: Stim folder {os.path.relpath(spikes_path, path)} for simulation /{os.path.split(path)[1]} already exists ... SKIPPING!')
        return {'stim_file': rel_stim_file, 'stim_name': cfg['stim_name']}
    
    print(f'INFO: Generating "{cfg["stim_name"]}" stimulus in folder {os.path.relpath(spikes_path, path)} for simulation /{os.path.split(path)[1]}!')

    ## Load SONATA circuit (using bluepysnap)
    circ = Circuit(cfg["circuit_config"])

    # _Step1_: Define stimuli (spatial structure)

    ## Get (virtual) fiber node IDs and locations
    stim_src = _find_popul_node_set(circ, cfg["proj_name"])  # Assuming there exists exactly one node set corresponding to given projection (=virtual source population)
    nids, pos2d, pos3d, dir3d = projloc.get_projection_locations(circ, cfg['proj_name'], cfg['proj_mask'], mask_type="dist")

    ## Cluster groups of nearby fibers (blobs) based on 3D locations [DON'T USE 2D LOCATIONS, since they may be discretized and contain duplicates due to flatmap conversion]
    grp_nids, grp_pos, grp_idx = projloc.cluster_by_locations(nids, pos3d, n_per_cluster=cfg['num_fibers_per_cluster'], n_clusters=cfg['n_clusters'])

    ## Plot groups of fibers
    _, pos2d_all, pos3d_all, _ = projloc.get_projection_locations(circ, cfg['proj_name'], None, None)
    projloc.plot_clusters_of_fibers(grp_idx, grp_pos, pos2d, pos3d, pos2d_all, pos3d_all, figs_path)

    ## Plot cluster size distribution
    projloc.plot_cluster_size_distribution(grp_idx, figs_path)

    ## Generate spatial patterns
    stimuli_seeds_nopt = cfg['stimuli_seeds_nopt']
    if stimuli_seeds_nopt is None or stimuli_seeds_nopt == 0:  # No spatial optimization
        stim_seeds = cfg['stimuli_seeds']
        min_cl_dist = None
    else:  # Optimize spatial distance over stimuli_seeds_nopt seeds
        min_dist_mat = np.zeros((stimuli_seeds_nopt, len(cfg['stimuli_seeds'])))
        for n_opt in range(stimuli_seeds_nopt):
            stim_seeds_tmp = [_seed + n_opt for _seed in cfg['stimuli_seeds']]
            pattern_grps_tmp, pattern_nids_tmp, pattern_pos2d_tmp, pattern_pos3d_tmp = stgen.generate_spatial_pattern(nids, pos2d, pos3d, grp_idx, stim_seeds_tmp, cfg['sparsity'])
            for _pidx in range(len(pattern_grps_tmp)):
                cl_centers = np.array(grp_pos)[np.array(pattern_grps_tmp[_pidx]), :]
                dmat = distance_matrix(cl_centers, cl_centers)
                min_dist_mat[n_opt, _pidx] = np.min(dmat[dmat > 0.0])  # Minimum distance between cluster centers
        opt_seed_offsets = list(np.argmax(min_dist_mat, 0))  # Choosing seed offsets (per pattern) with highest minimum distance between clusters
        min_cl_dist = np.max(min_dist_mat, 0)  # Min. cluster (center) distance per pattern
        stim_seeds = [_seed + _off for _seed, _off in zip(cfg['stimuli_seeds'], opt_seed_offsets)]
        print(f'SPATIAL STIMULUS OPTIMIZATION:\n  Seed offsets: {opt_seed_offsets}\n  Opt. seeds: {stim_seeds}\n  Min. cluster dist.: {min_cl_dist}')
    if len(np.unique(stim_seeds)) != len(stim_seeds):
        print('WARNING: Stimulus seeds not unique!')
    pattern_grps, pattern_nids, pattern_pos2d, pattern_pos3d = stgen.generate_spatial_pattern(nids, pos2d, pos3d, grp_idx, stim_seeds, cfg['sparsity'])

    ## Plot spatial patterns
    stgen.plot_spatial_patterns(pattern_pos2d, pattern_pos3d, pos2d, pos3d, pos2d_all, pos3d_all, figs_path)

    # _Step2_: Define stim series (stim train)

    ## Generate stimulus series (stim train)
    num_patterns = len(pattern_grps)
    stim_train = stgen.generate_stim_series(num_patterns, cfg['num_stimuli'], cfg['series_seed'], cfg['strict_enforce_p'], cfg['p_seeds'], cfg['overexpressed_tuples'])

    ## Plot stim train
    stgen.plot_stim_series(stim_train, figs_path)

    # _Step3_: Define analog rate signal (based on outputs of steps 1 & 2)

    ## Generate analog rate signals map (per group of fibers)
    num_groups = len(grp_nids)
    rate_map, time_axis, time_windows = stgen.generate_rate_signals(stim_train, pattern_grps, num_groups, cfg['start'], cfg['duration_stim'], cfg['duration_blank'], cfg['rate_min'], cfg['rate_max'])

    if time_windows[-1] > sim_duration:
        print('WARNING: Generated stimulus signals longer than simulation duration!')

    ## Plot analog rate signals
    stgen.plot_rate_signals(rate_map, time_axis, stim_train, time_windows, figs_path)

    # _Step4_: Define spike trains (temporal structure; based on output of step 3)

    ## Generate spikes for each group
    spike_map = stgen.generate_spikes(rate_map, time_axis, cfg['spike_seed'], cfg['bq'], cfg['tau'])

    ## Plot spikes for each group
    stgen.plot_spikes(spike_map, time_axis, stim_train, time_windows, 'Group idx', False, figs_path)

    ## Map groups to fibers
    out_map = stgen.map_groups_to_fibers(spike_map, grp_nids)

    ## Plot output spikes per fiber & stimulus PSTHs
    stgen.plot_spikes(out_map, time_axis, stim_train, time_windows, 'Virtual fiber node IDs', False, figs_path)
    stgen.plot_PSTHs(out_map, stim_train, time_windows, 10, figs_path)

    # _Step5_: Write spike output & properties files (based on output of step 4)

    ## Spike file, containing actual spike trains
    gid_out_map = {k + 1: v for k, v in out_map.items()}  # IMPORTANT: Convert SONATA node IDs (0-based) to NEURON cell IDs (1-based)!!
    stgen.write_spike_file(gid_out_map, stim_file)        # (See https://sonata-extension.readthedocs.io/en/latest/blueconfig-projection-example.html#dat-spike-files)

    ## Properties file, containing config params and properties of generated stimulus
    props = {'grp_nids': grp_nids, 'grp_pos': grp_pos, 'pattern_grps': pattern_grps,
             'stim_train': stim_train, 'time_windows': time_windows,
             'grp_idx': grp_idx, 'pos2d': pos2d, 'pos3d': pos3d, 'pos2d_all': pos2d_all,
             'pos3d_all': pos3d_all, 'pattern_pos2d': pattern_pos2d, 'pattern_pos3d': pattern_pos3d,
             'opt_stim_seeds': stim_seeds, 'opt_min_cl_dist': min_cl_dist}  # (Spatial pattern optimization)

    def np_conv(data):
        """ Convert numpy.ndarray to list recursively, so that JSON serializable """
        if isinstance(data, dict):
            return {k: np_conv(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [np_conv(d) for d in data]
        elif (isinstance(data, np.ndarray) or issubclass(data.__class__, np.number)) and hasattr(data, 'tolist'):
            return data.tolist()
        else:
            return data

    with open(props_file, 'w') as f:
        json.dump({'cfg': cfg, 'props': np_conv(props)}, f, indent=2)

    # Can be easily derived from these properties:
    # pattern_nids = [np.hstack([grp_nids[grp] for grp in grps]) for grps in pattern_grps]

    return {'stim_file': rel_stim_file, 'stim_src': stim_src, 'stim_name': cfg['stim_name']}


### [MODIFIED from A. Ecker's code base] ###
def _get_chached_projection_locations(path, c, proj_name, mask):
    """Local helper to avoid looping `projloc.get_projection_locations()` by saving the results
    and next time loading the saved results instead of recalculating the whole thing again"""
    save_name = os.path.join(os.path.split(path)[0], "_projections_cache_", "%s__%s.txt" % (proj_name, mask))
    if not os.path.isfile(save_name):
        nids, pos2d, pos3d, _ = projloc.get_projection_locations(c, proj_name, mask=mask, mask_type="dist")
        pos = pos2d if pos2d is not None else pos3d
        if not os.path.isdir(os.path.dirname(save_name)):
            os.mkdir(os.path.dirname(save_name))
        np.savetxt(save_name, np.concatenate((nids.reshape(-1, 1), pos), axis=1))
    else:
        tmp = np.loadtxt(save_name)
        nids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
    # return None, None
    return nids, pos


def _find_popul_node_set(c, popul_name):
    """Finds node set corresponding to given population"""
    nset = [k for k, v in c.node_sets.content.items() if "population" in v and v["population"] == popul_name]
    assert len(nset) == 1, f"ERROR: Node set corresponding to population '{popul_name}' not found!"
    return nset[0]


### [MODIFIED from A. Ecker's code base] ###
def generate_whisker_flick_stim(*, path, **kwargs):
    """Generates separate whisker flick-like VPM and POm spike trains"""
    param_list = ["circuit_config", # base circuit
                  "stim_seed", "stim_target", "vpm_pct", "vpm_proj_name", "pom_pct", "pom_proj_name",  # stim. structure
                  "sim_duration", "stim_delay", "stim_rate",  # stim. series
                  "vpm_mu", "pom_mu", "vpm_sigma", "pom_sigma", "vpm_spike_rate", "pom_spike_rate",
                  "index_shift"]  # spikes
    cfg = {p: kwargs.get(p) for p in param_list}
    stim_file_dict = {}

    # Load SONATA circuit (using bluepysnap)
    c = Circuit(cfg["circuit_config"])

    # Generate the stim times
    np.random.seed(cfg["stim_seed"])
    stim_times = spikewriter.generate_stim_series(cfg["stim_delay"], cfg["sim_duration"], cfg["stim_rate"])
    np.savetxt(os.path.join(path, "stim_times.txt"), stim_times, fmt="%f")  # Save stim times to .txt file

    # VPM spike train
    if cfg["vpm_proj_name"]:
        vpm_node_set = _find_popul_node_set(c, cfg["vpm_proj_name"])  # Assuming there exists exactly one node set corresponding to given projection (=virtual population)
        vpm_nids, _ = _get_chached_projection_locations(path, c, cfg["vpm_proj_name"], cfg["stim_target"])
        vpm_nids = np.sort(np.random.choice(vpm_nids, size=np.round(len(vpm_nids) * cfg["vpm_pct"] / 100.).astype(int), replace=False))
        # np.savetxt(os.path.join(path, "vpm_nids.txt"), vpm_nids, fmt="%d")  # Save selected fiber node IDs to .txt file
        if len(vpm_nids) > 0:
            # Generate the spike train
            vpm_spike_times, vpm_spiking_nids = spikewriter.generate_lognormal_spike_train(
                stim_times, vpm_nids, cfg["vpm_mu"], cfg["vpm_sigma"], cfg["vpm_spike_rate"], cfg["stim_seed"]
            )
        else:
            vpm_spike_times = []
            vpm_spiking_nids = []
        # Write to file and return file name used in the template
        vpm_stim_file = "vpm_input.dat"
        # vpm_spiking_gids = np.array(vpm_spiking_nids)
        vpm_spiking_gids = np.array(vpm_spiking_nids) + cfg["index_shift"]  # IMPORTANT: Convert SONATA node IDs (0-based) to NEURON cell IDs (1-based)!!
                                                           # (See https://sonata-extension.readthedocs.io/en/latest/blueconfig-projection-example.html#dat-spike-files)
        # spikewriter.write_spikes(vpm_spike_times, vpm_spiking_gids, os.path.join(path, vpm_stim_file))
        # stim_file_dict.update({"vpm_stim_file": vpm_stim_file, "vpm_node_set": vpm_node_set, "vpm_stim_src": cfg["vpm_proj_name"]})

        h5_vpm_stim_file = "vpm_input.h5"
        write_spike_file(vpm_spiking_gids, vpm_spike_times, cfg["vpm_proj_name"], os.path.join(path, h5_vpm_stim_file))
        stim_file_dict.update({"vpm_stim_file": h5_vpm_stim_file, "vpm_node_set": vpm_node_set, "vpm_stim_src": cfg["vpm_proj_name"]})

    # # POm spike train
    # if cfg["pom_proj_name"]:
    #     pom_node_set = _find_popul_node_set(c, cfg["pom_proj_name"])  # Assuming there exists exactly one node set corresponding to given projection (=virtual population)
    #     pom_nids, _ = _get_chached_projection_locations(path, c, cfg["pom_proj_name"], cfg["stim_target"])
    #     pom_nids = np.sort(np.random.choice(pom_nids, size=np.round(len(pom_nids) * cfg["pom_pct"] / 100.).astype(int), replace=False))
    #     np.savetxt(os.path.join(path, "pom_nids.txt"), pom_nids, fmt="%d")  # Save selected fiber node IDs to .txt file
    #     if len(pom_nids) > 0:
    #         # Generate the spike train
    #         pom_spike_times, pom_spiking_nids = spikewriter.generate_lognormal_spike_train(
    #             stim_times, pom_nids, cfg["pom_mu"], cfg["pom_sigma"], cfg["pom_spike_rate"], cfg["stim_seed"]
    #         )
    #     else:
    #         pom_spike_times = []
    #         pom_spiking_nids = []
    #     # Write to file and return file name used in the template
    #     pom_stim_file = "pom_input.dat"
    #     pom_spiking_gids = np.array(pom_spiking_nids) + 1  # IMPORTANT: Convert SONATA node IDs (0-based) to NEURON cell IDs (1-based)!!
    #                                                        # (See https://sonata-extension.readthedocs.io/en/latest/blueconfig-projection-example.html#dat-spike-files)
    #     spikewriter.write_spikes(pom_spike_times, pom_spiking_gids, os.path.join(path, pom_stim_file))
    #     stim_file_dict.update({"pom_stim_file": pom_stim_file, "pom_node_set": pom_node_set})

    return stim_file_dict

import pandas as pd
import h5py

def write_spike_file(gid_list, time_list, popul_name, out_file):
    """
    Writes SONATA output spike trains to file.
    
    Spike file format specs: https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#spike-file
    """
    # IMPORTANT: Convert SONATA node IDs (0-based) to NEURON cell IDs (1-based)!!
    # (See https://sonata-extension.readthedocs.io/en/latest/blueconfig-projection-example.html#dat-spike-files)
    # out_map = {k + 1: v for k, v in out_map.items()}

    out_path = os.path.split(out_file)[0]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # fext = os.path.splitext(out_file)[-1]
    # if fext == '.dat':
    #     with open(out_file, 'w') as f:
    #         f.write('/scatter\n')
    #         for gid, spike_times in out_map.items():
    #             if spike_times is not None:
    #                 for t in spike_times:
    #                     f.write(f'{t:f}\t{gid:d}\n')
    # elif fext == '.h5':
    # assert popul_name is not None, "ERROR: Population name required for '.h5' spike files."
    # time_list = []
    # gid_list = []
    # for gid, spike_times in out_map.items():
    #     if spike_times is not None:
    #         for t in spike_times:
    #             time_list.append(t)
    #             gid_list.append(gid)
    spike_df = pd.DataFrame(np.array([time_list, gid_list]).T, columns=['t', 'gid'])
    spike_df = spike_df.astype({'t': float, 'gid': int})
    spike_df.sort_values(by=['t', 'gid'], inplace=True)  # Sort by time
    with h5py.File(out_file, 'w') as f:
        pop = f.create_group(f"/spikes/{popul_name}")
        ts = pop.create_dataset("timestamps", data=spike_df['t'].values, dtype=np.float64)
        nodes = pop.create_dataset("node_ids", data=spike_df['gid'].values, dtype=np.uint64)
        ts.attrs['units'] = 'ms'
        # NOTE: Don't set optional 'sorting' attribute, since it will cause a warning:
        # .../highfive-2.10.0-eumanh/include/highfive/bits/H5ReadWrite_misc.hpp: 144 [WARN] sorting": data and hdf5 dataset have different types: Enum8 -> Integer8
        # pop.attrs['sorting'] = np.uint8(SimulationConfig.Output.SpikesSortOrder.by_time.value)

    # else:
    #     assert False, f"Output spike format '{fext}' not supported!"






### [MODIFIED from A. Ecker's code base] ###
def _generate_rnd_pattern_stim_series(path, stim_start, stim_end, stim_rate, pattern_names, seed):
    """Local helper to avoid looping `spikewriter.generate_rnd_pattern_stim_series()` by saving the results
    and next time loading the saved results instead of recalculating the whole thing again"""
    save_name = os.path.join(os.path.split(path)[0], "input_spikes",
                   "stimulus_stream__start%i__end%i__rate%i__seed%i.txt" % (stim_start, stim_end, stim_rate, seed))
    if not os.path.isfile(save_name):
        stim_times = spikewriter.generate_rnd_pattern_stim_series(stim_start, stim_end, stim_rate, pattern_names,
                                                                  seed, save_name, init_transient=None)
    else:
        stim_times = spikewriter.load_rnd_pattern_stim_series(save_name)
        # TODO: this should be handled more elegantly as the pattern names aren't part of the filename
        assert pattern_names == list(stim_times.keys()), "Loaded pattern names aren't the same ones as saved" \
                                                         "for the previous simulation in this campaign"
    return stim_times


### [MODIFIED from A. Ecker's code base] ###
def _get_projection_locations(path, c, proj_name, mask, supersample):
    """Local helper to avoid looping `projloc.get_projection_locations()` by saving the results
    and next time loading the saved results instead of recalculating the whole thing again"""
    supersample_str = "__supersample" if supersample else ""
    save_name = os.path.join(os.path.split(path)[0], "projections", "%s__%s%s.txt" % (proj_name, mask, supersample_str))
    if not os.path.isfile(save_name):
        nids, pos2d, pos3d, _ = projloc.get_projection_locations(c, proj_name, mask=mask,
                                                                 mask_type="dist", supersample=supersample)
        pos = pos2d if pos2d is not None else pos3d
        if not os.path.isdir(os.path.dirname(save_name)):
            os.mkdir(os.path.dirname(save_name))
        np.savetxt(save_name, np.concatenate((nids.reshape(-1, 1), pos), axis=1))
    else:
        tmp = np.loadtxt(save_name)
        nids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
    return nids, pos



