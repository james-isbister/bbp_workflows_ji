# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        C. Pokorny
# Date:          17/02/2021
# Last modified: 28/09/2021

import os
import shutil
import numpy as np
import pandas as pd
from bluepy import Circuit
import lookup_projection_locations as projloc
import stimulus_generation as stgen
import hashlib
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


""" Generates user target file, combining targets from custom list and projection paths """
def generate_user_target(*, circuit_config, path, custom_user_targets=[], **kwargs):
    
    circuit = Circuit(circuit_config)
    proj_paths = list(circuit.config['projections'].values())
    proj_targets = [os.path.join(os.path.split(p)[0], 'user.target') for p in proj_paths]
    proj_targets = list(filter(os.path.exists, proj_targets))
    
    target_paths = custom_user_targets + proj_targets
    
    user_target_name = 'user.target'
    user_target_file = os.path.join(path, user_target_name)
    with open(user_target_file, 'w') as f_tgt:
        for p in target_paths:
            assert os.path.exists(p), f'ERROR: Target "{p}" not found!'
            with open(p, 'r') as f_src:
                f_tgt.write(f_src.read())
                f_tgt.write('\n\n')
            # print(f'INFO: Adding target "{p}" to "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    # Set group membership to same as <path> (should be 10067/"bbp")
    # os.chown(user_target_file, uid=-1, gid=os.stat(path).st_gid)
    
    # print(f'INFO: Generated user target "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    return {'user_target_name': user_target_name}


""" Places stimulation file from template into simulation folders """
def stim_file_from_template(*, path, stim_file_template, **kwargs):
    
    stim_filename = 'input.dat'
    stim_file = os.path.join(path, stim_filename)
    shutil.copyfile(stim_file_template, stim_file)
    
    # print(f'INFO: Added stim file from template to {stim_file}')
    
    return {'stim_file': stim_filename}


""" Sets user-defined (e.g., layer-specific) depol levels based on scaling factors and method """
def apply_depol_scaling(*, depol_scale_method, depol_scale_factors, **kwargs):
    
    assert np.all([tgt in kwargs for tgt in depol_scale_method.keys()]), 'ERROR: Scale target error!'
    
    # Scaling function, depending on method
    # [Comparison of methods: git@bbpgitlab.epfl.ch/conn/structural/dendritic_synapse_density/MissingSynapses.ipynb]
    def scale_fct(value, scale, method):
        if method == 'none': # No scaling
            scaled_value = value
        elif method == 'linear': # Linear scaling
            scaled_value = value * scale
        elif method == 'linear_bounded': # Linear scaling, bounded to max=100.0
            max_value = 100.0
            scaled_value = np.minimum(value * scale, max_value)
        elif method == 'exponential': # Exponential scaling, converging to max=100.0
            max_value = 100.0
            tau = -1 / np.log(1 - value / max_value)
            scaled_value = max_value * (1 - np.exp(-scale / tau))
        else:
            assert False, 'ERROR: Scale method unknown!'
        # return np.round(scaled_value).astype(int)
        return scaled_value
    
    # Apply scaling
    depol_scale_dict = {}
    for spec, scale in depol_scale_factors.items(): # Specifier and scaling factor, e.g. "L1I": 0.6
        for tgt in depol_scale_method.keys(): # Scaling target, e.g. "depol_mean_pct"
            depol_scale_dict.update({f'{tgt}_{spec}': scale_fct(kwargs[tgt], scale, depol_scale_method[tgt])})    
    
    return depol_scale_dict


def fake_stim_file(*, path, stim_file_template, **kwargs):
    return {'stim_file': stim_file_template}



def set_conductance_scalings_for_unconnected_frs(*, depol_stdev_mean_ratio, fr_scale, in_vivo_reference_frs, **kwargs):

    scaling_and_data_neuron_class_keys = {
    
    "L1I":"L1_INH",
    "L23E":"L23_EXC",
    "L23I":"L23_INH",
    "L4E":"L4_EXC",
    "L4I":"L4_INH",
    "L5E":"L5_EXC",
    "L5I":"L5_INH",
    "L6E":"L6_EXC",
    "L6I":"L6_INH"
    
    }

    scale_dict = {}
    for scaling_neuron_class_key, in_vivo_fr in in_vivo_reference_frs.items():
        # print(neuron_class_key, in_vivo_fr)
        

        data_for_unconnected_fit = pd.read_parquet(path="data_for_unconnected_fit.parquet")
        ng_data_for_plot = data_for_unconnected_fit[(data_for_unconnected_fit["ca"] == 1.15) & (data_for_unconnected_fit["neuron_class"] == scaling_and_data_neuron_class_keys[scaling_neuron_class_key])]
        gradient_line_x = np.linspace(0,41,1000)
        gradient_line_y = gradient_line_x * depol_stdev_mean_ratio
        predicted_frs_for_line = griddata(ng_data_for_plot[['mean', 'stdev']].to_numpy(), ng_data_for_plot["data"], (gradient_line_x, gradient_line_y), method='cubic')

        in_vivo_fr = in_vivo_reference_frs[scaling_neuron_class_key]

        scaled_target_fr = fr_scale*in_vivo_fr
        
        index_of_closest_point = np.nanargmin(abs(predicted_frs_for_line - scaled_target_fr))
        closest_x = round(gradient_line_x[index_of_closest_point], 3)
        closest_y = round(gradient_line_y[index_of_closest_point], 3)
        # closest_fr = round(predicted_frs_for_line[index_of_closest_point], 3)

        scale_dict.update({f'predicted_fr_{scaling_neuron_class_key}': scaled_target_fr})
        scale_dict.update({f'ornstein_uhlenbeck_mean_pct_{scaling_neuron_class_key}': closest_x})
        scale_dict.update({f'ornstein_uhlenbeck_sd_pct_{scaling_neuron_class_key}': closest_y})

    return scale_dict


def get_cfg_hash(cfg):
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





import spikewriter

# Originally created by Andras Ecker
def _load_circuit_targets(circuit_path, user_target_path):
    """Loads circuit and adds targets from user defined user.target"""
    c = Circuit(circuit_path)
    if os.path.exists(user_target_path):
        c_cfg = c.config.copy()
        c_cfg["targets"].append(user_target_path)
        c = Circuit(c_cfg)  # reload circuit with extra targets from user.target
    return c


# Originally created by Andras Ecker
def gen_whisker_flick_stim(*, path, **kwargs):
    """Generates whisker flick like VPM and optionally POm (on top) spike trains
    (Hard coded to Bio_M hex_O1 geometry)"""
    param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
                  "stim_seed", "vpm_pct", "pom_pct",  # structural
                  "sim_duration", "stim_delay", "stim_rate",  # stim. series
                  "vpm_mu", "pom_mu", "vpm_sigma", "pom_sigma", "vpm_spike_rate", "pom_spike_rate"]  # spikes
    cfg = {p: kwargs.get(p) for p in param_list}
    assert cfg["circuit_target"] == "hex_O1", "Projection gids are hard coded to (Bio_M) hex_O1." \
                                              "Please change the target of the simulation or adapt the code!"
    # Load circuit and fix targets
    c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])
    # VPM spike train
    gids = c.cells.ids("proj_Thalamocortical_VPM_Source_hex_O1")  # user.target should have this hard coded name
    np.random.seed(cfg["stim_seed"])
    gids = np.sort(np.random.choice(gids, size=int(len(gids) * cfg["vpm_pct"]/100.), replace=False))
    stim_times = spikewriter.generate_stim_series(cfg["stim_delay"], cfg["sim_duration"], cfg["stim_rate"])
    spike_times, spiking_gids = spikewriter.generate_lognormal_spike_train(stim_times, gids,
                                            cfg["vpm_mu"], cfg["vpm_sigma"], cfg["vpm_spike_rate"], cfg["stim_seed"])
    # POm spike train
    if cfg["pom_pct"] > 0.0:
        pom_gids = c.cells.ids("proj_Thalamocortical_POM_Source_hex_O1")  # user.target should have this hard coded name
        np.random.seed(cfg["stim_seed"])
        pom_gids = np.sort(np.random.choice(pom_gids, size=int(len(pom_gids) * cfg["pom_pct"]/100.), replace=False))
        pom_spike_times, pom_spiking_gids = spikewriter.generate_lognormal_spike_train(stim_times, pom_gids,
                                                        cfg["pom_mu"], cfg["pom_sigma"],
                                                        cfg["pom_spike_rate"], cfg["stim_seed"])
        spike_times, spiking_gids = spikewriter.merge_spike_trains([spike_times, pom_spike_times],
                                                                   [spiking_gids ,pom_spiking_gids])
    # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(spike_times, spiking_gids, os.path.join(path, stim_file))
    return {"stim_file": stim_file}

def _get_projection_locations(path, c, proj_name, mask, supersample):
    """Local helper to avoid looping `projloc.get_projection_locations()` by saving the results
    and next time loading the saved results instead of recalculating the whole thing again"""
    supersample_str = "__supersample" if supersample else ""
    save_name = os.path.join(os.path.split(path)[0], "projections", "%s__%s%s.txt" % (proj_name, mask, supersample_str))
    if not os.path.isfile(save_name):
        gids, pos2d, pos3d, _ = projloc.get_projection_locations(c, proj_name, mask=mask,
                                                                 mask_type="dist", supersample=supersample)
        pos = pos2d if pos2d is not None else pos3d
        if not os.path.isdir(os.path.dirname(save_name)):
            os.mkdir(os.path.dirname(save_name))
        np.savetxt(save_name, np.concatenate((gids.reshape(-1, 1), pos), axis=1))
    else:
        tmp = np.loadtxt(save_name)
        gids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
    return gids, pos


# Adapted from code created by Andras Ecker
def gen_whisker_flick_stim_and_find_fibers(*, path, **kwargs):
    """
    Generates whisker step (longer then flick) like VPM and optionally POm (on top) spike trains
    VPM fibers are clustered together (to e.g. scan toposample stim. params.) and as clustering happens on the fly,
    it's a more general version of `gen_whisker_flick_stim()` that can be applied to any version and region of the SSCx
    """

    param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
                  "stim_seed", "vpm_pct", "pom_pct", # structural
                  "stim_delay", "num_stims", "inter_stimulus_interval",  # stim. series
                  "vpm_mu", "pom_mu", "vpm_sigma", "pom_sigma", "vpm_spike_rate", "pom_spike_rate",
                  "vpm_proj_name", "pom_proj_name", "supersample"]  # spikes

    cfg = {p: kwargs.get(p) for p in param_list}
    # Load circuit and fix targets
    c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])

    all_spike_times = []; all_spiking_gids = []
    fib_grps = ['vpm', 'pom']
    for fib_grp_i, fib_grp in enumerate(fib_grps):

        gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], cfg["circuit_target"], cfg["supersample"])
        np.random.seed(cfg["stim_seed"] + fib_grp_i)
        selected_gid_indices = np.sort(np.random.choice(range(len(gids)), size=int(len(gids) * cfg[fib_grp + "_pct"]/100.), replace=False))
        selected_gids = gids[selected_gid_indices]
        selected_gid_poss = pos[selected_gid_indices]

        plt.figure()
        plt.scatter(np.asarray(pos)[:, 0], np.asarray(pos)[:, 1])
        plt.scatter(np.asarray(selected_gid_poss)[:, 0], np.asarray(selected_gid_poss)[:, 1])
        plt.savefig(os.path.join(path, fib_grp + '_selected_stim_fibers.png'))
        plt.close()

        stim_times = spikewriter.generate_stim_series_num_stims(cfg["stim_delay"], cfg["num_stims"], cfg["inter_stimulus_interval"])

        spike_times, spiking_gid_indices = spikewriter.generate_lognormal_spike_train(stim_times, selected_gid_indices,
                                                cfg[fib_grp + "_mu"], cfg[fib_grp + "_sigma"], cfg[fib_grp + "_spike_rate"], cfg["stim_seed"] + fib_grp_i)



        spiking_gids = gids[spiking_gid_indices]


        plt.figure()
        plt.scatter(spike_times, spiking_gid_indices)
        plt.gca().set_xlim([2000, 2100])
        plt.savefig(os.path.join(path, fib_grp + 'stim_spikes.png'))
        plt.close()

        all_spike_times += spike_times.tolist()
        all_spiking_gids += spiking_gids.tolist()

        # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(all_spike_times, all_spiking_gids, os.path.join(path, stim_file))

    return {"stim_file": stim_file}




# def generate_boundle_whisker_step_stim(*, path, **kwargs):
#     """
#     Generates whisker step (longer then flick) like VPM and optionally POm (on top) spike trains
#     VPM fibers are clustered together (to e.g. scan toposample stim. params.) and as clustering happens on the fly,
#     it's a more general version of `gen_whisker_flick_stim()` that can be applied to any version and region of the SSCx
#     """
#     param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
#                   "stim_seed", "vpm_pct", "vpm_proj_name", "pom_pct", "pom_proj_name", "supersample", "n_clusters",  # structural
#                   "reconn_delay", "sim_duration", "stim_delay", "stim_rate",  # stim. series
#                   "stim_duration", "min_rate", "vpm_max_rate",  # VPM rate (POm rate is derived from these)
#                   "bq", "tau"]  # spikes
#     cfg = {p: kwargs.get(p) for p in param_list}
#     # Load circuit and fix targets
#     c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])
#     # VPM spike train
#     # Get spatial location of fibers and cluster them to "bundles"
#     gids, pos = _get_projection_locations(path, c, cfg["vpm_proj_name"], cfg["circuit_target"], cfg["supersample"])
#     grp_gids, _, grp_idx = projloc.cluster_by_locations(gids, pos, n_clusters=cfg["n_clusters"])
#     # Randomly select a given pct of bundles (preselect the ones with average size)
#     len_grps = np.array([len(grp) for grp in grp_gids])
#     m_lens, std_lens = np.mean(len_grps), np.std(len_grps)
#     viable_grp_idx = np.where((len_grps > m_lens-std_lens) & (len_grps < m_lens+std_lens))[0]
#     np.random.seed(cfg["stim_seed"])
#     pattern_idx = np.random.choice(viable_grp_idx, size=int(cfg["n_clusters"] * cfg["vpm_pct"]/100.), replace=False)
#     pattern_gids = np.sort(np.concatenate([gids[grp_idx == grp] for grp in pattern_idx]))
#     # Generate the spike trains
#     stim_times = spikewriter.generate_stim_series(cfg["stim_delay"], cfg["sim_duration"], cfg["stim_rate"])
#     t, stim_rate = spikewriter.generate_rate_signal(cfg["reconn_delay"], cfg["sim_duration"], stim_times,
#                                                     cfg["stim_duration"], cfg["min_rate"], cfg["vpm_max_rate"])
#     spike_times, spiking_gids = spikewriter.generate_inh_adaptingmarkov_spike_train(pattern_gids, t, stim_rate,
#                                             cfg["bq"], cfg["tau"], cfg["stim_seed"])
#     # POm spike train (no clustering of fibers here)
#     if cfg["pom_pct"] > 0.0:
#         pom_max_rate = 0.5 * cfg["vpm_max_rate"]
#         pom_gids, _ = _get_projection_locations(path, c, cfg["pom_proj_name"], cfg["circuit_target"], False)
#         np.random.seed(cfg["stim_seed"])
#         pom_gids = np.sort(np.random.choice(pom_gids, size=int(len(pom_gids) * cfg["pom_pct"]/100.), replace=False))
#         _, pom_rate = spikewriter.generate_rate_signal(cfg["reconn_delay"], cfg["sim_duration"], stim_times,
#                                                        cfg["stim_duration"], cfg["min_rate"], pom_max_rate)
#         pom_spike_times, pom_spiking_gids = spikewriter.generate_inh_adaptingmarkov_spike_train(pom_gids, t, pom_rate,
#                                                         cfg["bq"], cfg["tau"], cfg["stim_seed"])
#         spike_times, spiking_gids = spikewriter.merge_spike_trains([spike_times, pom_spike_times],
#                                                                    [spiking_gids ,pom_spiking_gids])
#     # Write to file and return file name used in the template
#     stim_file = "input.dat"
#     spikewriter.write_spikes(spike_times, spiking_gids, os.path.join(path, stim_file))
#     return {"stim_file": stim_file}



# def generate_random_dot_flash_stimulus(*, path, sim_duration, **kwargs):
#     """
#     Generates 'random dot flash' type of stimulus file and writing the
#     spikes file(s) to hashed folders to prevent multiple generation of
#     exact same spike files
#     """

#     # _Init_

#     ## Get stim config parameters and hash code
#     param_list = ['circuit_config', # Circuit
#                   'proj_name', 'proj_mask', 'proj_mask_type', 'proj_flatmap', 'num_fibers_per_cluster', 'stimuli_seeds', 'sparsity', # Spatial params
#                   'num_stimuli', 'series_seed', 'strict_enforce_p', 'p_seeds', 'overexpressed_tuples', # Series params
#                   'start', 'duration_stim', 'duration_blank', 'rate_min', 'rate_max', # Rate signal params
#                   'spike_seed', 'bq', 'tau'] # Spike params

#     cfg = {'stim_name': 'RandomDotFlash'}
#     cfg.update({p: kwargs.get(p) for p in param_list}) # Stim config parameters
#     cfg_hash = get_cfg_hash(cfg)

#     ## Define paths and files
#     spikes_path = os.path.join(os.path.split(path)[0], 'spikes', cfg_hash)
#     figs_path = os.path.join(spikes_path, 'figs')
#     stim_file = os.path.join(spikes_path, 'input.dat')
#     rel_stim_file = os.path.relpath(stim_file, path) # Relative path to current simulation folder
#     props_file = os.path.splitext(stim_file)[0] + '.json'

#     ## Check if stimulus folder for given parameter configuration already exists (using hash code as folder name)
#     ## [IMPORTANT: Thread-safe implementation w/o os.path.exists followed by os.makedirs, since this would create
#     ##             a race condition in case max_workers > 1 parallel processes are used!]
#     try:
#         os.makedirs(spikes_path)
#         # os.chown(spikes_path, uid=-1, gid=os.stat(path).st_gid) # Set group membership same as <path> (should be 10067/"bbp")

#         os.makedirs(figs_path)
#         # os.chown(figs_path, uid=-1, gid=os.stat(path).st_gid) # Set group membership same as <path> (should be 10067/"bbp")

#     except FileExistsError: # Stimulus folder already generated, stop here!
#         print(f'INFO: Stim folder {os.path.relpath(spikes_path, path)} for simulation /{os.path.split(path)[1]} already exists ... SKIPPING!')
#         return {'stim_file': rel_stim_file, 'stim_name': cfg['stim_name']}
    
#     print(f'INFO: Generating "{cfg["stim_name"]}" stimulus in folder {os.path.relpath(spikes_path, path)} for simulation /{os.path.split(path)[1]}!')

#     ## Load circuit
#     circ = Circuit(cfg['circuit_config'])

#     user_target_name = kwargs.get('user_target_name', '')
#     user_target_file = os.path.join(path, user_target_name)
#     if len(user_target_name) > 0 and os.path.exists(user_target_file):
#         # Make individual user targets generated by generate_user_target available
#         circ_cfg_dict = circ.config.copy()
#         circ_cfg_dict['targets'].append(user_target_file) # Add user target file to list of existing target files
#         circ = Circuit(circ_cfg_dict) # Re-load circuit

#     # print(f'INFO: Loaded circuit with {len(circ.cells.targets)} targets!')

#     # _Step1_: Define stimuli (spatial structure)

#     ## Get fiber GIDs and locations
#     gids, pos2d, pos3d, dir3d = projloc.get_projection_locations(circ, cfg['proj_name'], cfg['proj_mask'], cfg['proj_mask_type'], cfg['proj_flatmap'])

#     ## Cluster groups of nearby fibers (blobs) based on 3D locations [DON'T USE 2D LOCATIONS, since they may be discretized and contain duplicates due to flatmap conversion]
#     grp_gids, grp_pos, grp_idx = projloc.cluster_by_locations(gids, pos3d, n_per_cluster=cfg['num_fibers_per_cluster'])

#     ## Plot groups of fibers
#     _, pos2d_all, pos3d_all, _ = projloc.get_projection_locations(circ, cfg['proj_name'], None, None, cfg['proj_flatmap'])
#     projloc.plot_clusters_of_fibers(grp_idx, grp_pos, pos2d, pos3d, pos2d_all, pos3d_all, figs_path)

#     ## Plot cluster size distribution
#     projloc.plot_cluster_size_distribution(grp_idx, figs_path)

#     ## Generate spatial patterns
#     pattern_grps, pattern_gids, pattern_pos2d, pattern_pos3d = stgen.generate_spatial_pattern(gids, pos2d, pos3d, grp_idx, cfg['stimuli_seeds'], cfg['sparsity'])

#     ## Plot spatial patterns
#     stgen.plot_spatial_patterns(pattern_pos2d, pattern_pos3d, pos2d, pos3d, pos2d_all, pos3d_all, figs_path)

#     # _Step2_: Define stim series (stim train)

#     ## Generate stimulus series (stim train)
#     num_patterns = len(pattern_grps)
#     stim_train = stgen.generate_stim_series(num_patterns, cfg['num_stimuli'], cfg['series_seed'], cfg['strict_enforce_p'], cfg['p_seeds'], cfg['overexpressed_tuples'])

#     ## Plot stim train
#     stgen.plot_stim_series(stim_train, figs_path)

#     # _Step3_: Define analog rate signal (based on outputs of steps 1 & 2)

#     ## Generate analog rate signals map (per group of fibers)
#     num_groups = len(grp_gids)
#     rate_map, time_axis, time_windows = stgen.generate_rate_signals(stim_train, pattern_grps, num_groups, cfg['start'], cfg['duration_stim'], cfg['duration_blank'], cfg['rate_min'], cfg['rate_max'])

#     if time_windows[-1] > sim_duration:
#         print('WARNING: Generated stimulus signals longer than simulation duration!')

#     ## Plot analog rate signals
#     stgen.plot_rate_signals(rate_map, time_axis, stim_train, time_windows, figs_path)

#     # _Step4_: Define spike trains (temporal structure; based on output of step 3)

#     ## Generate spikes for each group
#     spike_map = stgen.generate_spikes(rate_map, time_axis, cfg['spike_seed'], cfg['bq'], cfg['tau'])

#     ## Plot spikes for each group
#     stgen.plot_spikes(spike_map, time_axis, stim_train, time_windows, 'Group idx', False, figs_path)

#     ## Map groups to fibers
#     out_map = stgen.map_groups_to_fibers(spike_map, grp_gids)

#     ## Plot output spikes per fiber & stimulus PSTHs
#     stgen.plot_spikes(out_map, time_axis, stim_train, time_windows, 'Fiber GIDs', False, figs_path)
#     stgen.plot_PSTHs(out_map, stim_train, time_windows, 10, figs_path)

#     # _Step5_: Write spike output & properties files (based on output of step 4)
    
#     ## Spike file, containing actual spike trains
#     stgen.write_spike_file(out_map, stim_file)
    
#     ## Properties file, containing config params and properties of generated stimulus
#     props = {'grp_gids': grp_gids, 'grp_pos': grp_pos, 'pattern_grps': pattern_grps,
#              'stim_train': stim_train, 'time_windows': time_windows}

#     def np_conv(data):
#         """ Convert numpy.ndarray to list recursively, so that JSON serializable """
#         if isinstance(data, dict):
#             return {k: np_conv(v) for k, v in data.items()}
#         elif isinstance(data, list):
#             return [np_conv(d) for d in data]
#         elif isinstance(data, np.ndarray):
#             return data.tolist()
#         else:
#             return data
    
#     with open(props_file, 'w') as f:
#         json.dump({'cfg': cfg, 'props': np_conv(props)}, f, indent=2)

#     # Can be easily derived from these properties:
#     # pattern_gids = [np.hstack([grp_gids[grp] for grp in grps]) for grps in pattern_grps]

#     return {'stim_file': rel_stim_file, 'stim_name': cfg['stim_name']}
