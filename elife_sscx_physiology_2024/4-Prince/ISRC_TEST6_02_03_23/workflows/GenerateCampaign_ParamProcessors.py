# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        P.Litvak and J.Isbister, adapted from C. Pokorny
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
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress
import pick_postsyn_targets as picker
import generate_stimuli as gs


""" Selects neurons for stimulation, finds their postsynapric targets and generates synchronous or rate coded spike trains. """
def rate_or_synchronous_stimulation_for_group_of_single_neurons(*, path, **kwargs):
    
    target_file_name = os.path.join(path, "postsynaptic_targets")
    circ = Circuit(kwargs['circuit_config'])
    circ_cfg_dict = circ.config.copy()
    for user_target_file in kwargs['custom_user_targets']:
        circ_cfg_dict['targets'].append(user_target_file) # Add user target file to list of existing target files
    circ = Circuit(circ_cfg_dict) # Re-load circuit

    #stimulus_gids = picker.create_postsyn_targets(circ, 'hex0', kwargs['n_stimulus_neurons'], -1, target_file_name) 
    stimulus_gids = picker.sample_presyn_targets_L23(circ, 'hex0', 'PYR', kwargs['n_stimulus_neurons'])
    stimulus_gids_file_name = os.path.join(path, "stimulus_gids")
    np.save(stimulus_gids_file_name, stimulus_gids)    
    
    experiment_dict = {k:kwargs[k] for k in ['experiment_type', 'up_rate', 'down_rate', 'stimulation_window', 'clip_width_range']}
    # experiment_dict = {k:kwargs[k] for k in ['experiment_type', 'up_and_down_rates', 'stimulation_window', 'clip_width_range']}
    experiment_dict['up_and_down_rates'] = [kwargs['down_rate'], kwargs['up_rate']]

    clip_states, clip_start_and_end_times = gs.generate_1_bit_pattern(experiment_dict, os.path.join(path, "state_times.dat"), kwargs['pulse_window'])
    spike_times_by_i, spike_ids_by_i = gs.generate_spikes_from_1_bit_pattern(experiment_dict, clip_states, clip_start_and_end_times, stimulus_gids, kwargs['pulse_window'])

    opto_sections = ''
    for i, stimulus_gid in enumerate(stimulus_gids):        
        stim_name = f'spike-stim_{i}_{stimulus_gid}'
        spike_times_file = os.path.join(path, f"opto_stim_spikes/pyramidal_gid_{stimulus_gid}_opto_spikes.dat")
        stim_dict = {'Mode': 'Current', 'Delay': 0, 'Duration': kwargs['sim_duration'], 'Pattern': 'SynapseReplay', 'SpikeFile': spike_times_file} # 'Delay': opto_t[stim_idx], 'Duration': opto_dur[stim_idx],
        opto_sections += config_section_from_dict('Stimulus ' + stim_name, stim_dict)

        inj_name = stim_name + '_inject'
        #inj_dict = {'Stimulus': stim_name, 'Target': f'pyramidal_gid_{stimulus_gid}'}
        inj_dict = {'Stimulus': stim_name, 'Target': f'Mosaic'}
        opto_sections += config_section_from_dict('StimulusInject ' + inj_name, inj_dict)

        gs.write_spikes(spike_times_by_i[i], spike_ids_by_i[i], spike_times_file)
        
    plt.figure()
    plt.scatter([item for sublist in spike_times_by_i for item in sublist], [item for sublist in spike_ids_by_i for item in sublist])
    plt.gca().set_xlim([0, kwargs['sim_duration']])
    plt.savefig(path + "/InputSpikesRaster.pdf")
    plt.close()

    return {'single_neuron_opto_spike_stims': opto_sections}


def config_section_from_dict(sect_name, param_dict, intend=4):
    """
    Generates a BlueConfig section string from a dict
    """

    section_str = sect_name + '\n{\n'
    for k, v in param_dict.items():
        section_str += ' ' * intend + f'{k} {str(v)}\n' 
    section_str += '}\n\n'
    return section_str


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


""" Sets user-defined (e.g., layer-specific) shot noise levels based on scaling factors and method """
def apply_shot_noise_scaling(*, shotn_scale_method, shotn_scale_factors, **kwargs):
    
    assert np.all([tgt in kwargs for tgt in shotn_scale_method.keys()]), 'ERROR: Scale target error!'
    
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
    shotn_scale_dict = {}
    for spec, scale in shotn_scale_factors.items(): # Specifier and scaling factor, e.g. "L1I": 0.6
        for tgt in shotn_scale_method.keys(): # Scaling target, e.g. "shotn_mean_pct"
            shotn_scale_dict.update({f'{tgt}_{spec}': scale_fct(kwargs[tgt], scale, shotn_scale_method[tgt])})    
    
    return shotn_scale_dict


def calculate_suggested_unconnected_firing_rate(target_connected_fr, a, b, c):

    y = target_connected_fr
    log_domain = max((y - c) / a, 1.0)
    # print(y, c, a, log_domain)
    suggested_unconnected_fr = math.log(log_domain) / b

    return suggested_unconnected_fr



def fit_exponential(ca_stat1, ca_stat2):

    popt, pcov = curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c,
        ca_stat1, ca_stat2, p0=(1.0, 0.5, ca_stat2.min() - 1), 
        maxfev=20000
    )

    return popt



def create_delete_flag(path):

    delete_flag_file = os.path.join(path, 'DELETE_FLAG.FLAG')
    f = open(delete_flag_file, "x")
    f.close()



def set_conductance_scalings_for_desired_frs(*, path, ca, depol_stdev_mean_ratio, desired_connected_proportion_of_invivo_frs, in_vivo_reference_frs, data_for_unconnected_fit_name, data_for_connected_adjustment_fit_name, unconnected_connected_fr_adjustment_fit_method, **kwargs):

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

    should_create_delete_flag = False
    
    unconn_df = pd.read_parquet(path=data_for_unconnected_fit_name)

    if (data_for_connected_adjustment_fit_name != ''):
        unconn_conn_df = pd.read_parquet(path=data_for_connected_adjustment_fit_name)
        # ca_unconn_conn_df = unconn_conn_df.etl.q(ca=ca, window='conn_spont')
        ca_unconn_conn_df = unconn_conn_df[(unconn_conn_df['ca']==ca) & (unconn_conn_df['depol_stdev_mean_ratio']==depol_stdev_mean_ratio) & (unconn_conn_df["window"] == 'conn_spont') & (unconn_conn_df['bursting'] == False)]

        ca_unconn_conn_df["invivo_fr"] = ca_unconn_conn_df['desired_connected_fr'] / ca_unconn_conn_df['desired_connected_proportion_of_invivo_frs']
        ca_unconn_conn_df = ca_unconn_conn_df[ca_unconn_conn_df['mean_of_mean_firing_rates_per_second'] < ca_unconn_conn_df['invivo_fr'] * 1.05]


    scale_dict = {}
    for scaling_neuron_class_key, in_vivo_fr in in_vivo_reference_frs.items():

        neuron_class = scaling_and_data_neuron_class_keys[scaling_neuron_class_key]

        in_vivo_fr = in_vivo_reference_frs[scaling_neuron_class_key]
        desired_connected_fr = desired_connected_proportion_of_invivo_frs*in_vivo_fr

        if (data_for_connected_adjustment_fit_name != ''):

            nc_ca_unconn_conn_df = ca_unconn_conn_df[ca_unconn_conn_df['neuron_class']==neuron_class]

            if (nc_ca_unconn_conn_df[nc_ca_unconn_conn_df["depol_stdev_mean_ratio"] == depol_stdev_mean_ratio]['mean_of_mean_firing_rates_per_second'].max() < desired_connected_fr):
                should_create_delete_flag = True

            unconnected_connected_fr_adjustment_fit_method = 'exponential'

            if (unconnected_connected_fr_adjustment_fit_method == 'exponential'):            

                popt = fit_exponential(nc_ca_unconn_conn_df['desired_unconnected_fr'], nc_ca_unconn_conn_df['mean_of_mean_firing_rates_per_second'])
                desired_unconnected_fr = calculate_suggested_unconnected_firing_rate(desired_connected_fr, popt[0], popt[1], popt[2])

            elif (unconnected_connected_fr_adjustment_fit_method == 'linear'):

                ca_lr = linregress(ca_stat1, ca_stat2)
                ca_lr_slope = np.around(ca_lr.slope, 3)

                desired_unconnected_fr = nc_ca_unconn_conn_df['desired_unconnected_fr'] / ca_lr_slope

            print(ca, neuron_class, desired_connected_fr, desired_unconnected_fr)
        
        else: 
            desired_unconnected_fr = desired_connected_fr



        nc_unconn_df = unconn_df[(unconn_df["ca"] == 1.15) & (unconn_df["neuron_class"] == scaling_and_data_neuron_class_keys[scaling_neuron_class_key])]
        
        # print(ng_data_for_plot)
        gradient_line_x = np.linspace(0,100,1000)
        gradient_line_y = gradient_line_x * depol_stdev_mean_ratio

        predicted_frs_for_line = griddata(nc_unconn_df[['mean', 'stdev']].to_numpy(), nc_unconn_df["data"], (gradient_line_x, gradient_line_y), method='cubic')
        
        # print(scaling_neuron_class_key, predicted_frs_for_line, desired_unconnected_fr)
        index_of_closest_point = np.nanargmin(abs(predicted_frs_for_line - desired_unconnected_fr))

        closest_x = round(gradient_line_x[index_of_closest_point], 3)
        closest_y = round(gradient_line_y[index_of_closest_point], 3)

        scale_dict.update({f'desired_connected_fr_{scaling_neuron_class_key}': desired_connected_fr})
        scale_dict.update({f'desired_unconnected_fr_{scaling_neuron_class_key}': desired_unconnected_fr})
        scale_dict.update({f'ornstein_uhlenbeck_mean_pct_{scaling_neuron_class_key}': closest_x})
        scale_dict.update({f'ornstein_uhlenbeck_sd_pct_{scaling_neuron_class_key}': closest_y})
        # scale_dict.update({f'shotn_mean_pct_{scaling_neuron_class_key}': closest_x})
        # scale_dict.update({f'shotn_sd_pct_{scaling_neuron_class_key}': closest_y})

        # print(scale_dict)

    if (should_create_delete_flag):
        create_delete_flag(path)

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
    if cfg["vpm_pct"] > 0.0:
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
        # print(save_name, tmp)
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
                  "vpm_proj_name", "pom_proj_name", "supersample",  # spikes
                  "ji_total_clusters", "ji_clusters_per_stimulus"]

    cfg = {p: kwargs.get(p) for p in param_list}
    # Load circuit and fix targets
    c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])

    all_spike_times = []; all_spiking_gids = []
    fib_grps = ['vpm', 'pom']
    for fib_grp_i, fib_grp in enumerate(fib_grps):

        # gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], cfg["circuit_target"], cfg["supersample"])
        # gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], "hex0", cfg["supersample"])
        gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], "cyl_hex0_0.010", cfg["supersample"])
        

        if (fib_grp == "vpm"):

            grp_gids, grp_pos, grp_idx = projloc.cluster_by_locations(gids, pos, n_clusters=cfg["ji_total_clusters"])
            new_pos = []; new_gids = [];

            plt.figure()
            for clus_i, clus_gids in enumerate(grp_gids):

                if (clus_i == cfg["ji_clusters_per_stimulus"]):
                    break

                indices = np.where(np.in1d(gids, clus_gids))[0]
                clus_gids = gids[indices]
                clus_pos = pos[indices, :]
                new_gids.extend(clus_gids)
                new_pos.extend(clus_pos)
                
                plt.scatter(clus_pos[:, 0], clus_pos[:, 1])

            plt.savefig(os.path.join(path, fib_grp + '_CLUSTER_stim_fibers.png'))
            plt.close()

            gids = np.asarray(new_gids)
            pos = np.asarray(new_pos)

        np.random.seed(cfg["stim_seed"] + fib_grp_i)

        pct_key = fib_grp + "_pct"

        if (cfg[pct_key] > 0.0):
            selected_gid_indices = np.sort(np.random.choice(range(len(gids)), size=int(len(gids) * cfg[pct_key]/100.), replace=False))
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
            plt.gca().set_xlim([1500, 1510])
            plt.savefig(os.path.join(path, fib_grp + 'stim_spikes.png'))
            plt.close()

            all_spike_times += spike_times.tolist()
            all_spiking_gids += spiking_gids.tolist()

        # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(all_spike_times, all_spiking_gids, os.path.join(path, stim_file))

    return {"stim_file": stim_file}


# Adapted from code created by Andras Ecker
def gen_whisker_flick_stim_and_find_fibers_all(*, path, **kwargs):
    """
    Generates whisker step (longer then flick) like VPM and optionally POm (on top) spike trains
    VPM fibers are clustered together (to e.g. scan toposample stim. params.) and as clustering happens on the fly,
    it's a more general version of `gen_whisker_flick_stim()` that can be applied to any version and region of the SSCx
    """

    param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
                  "stim_seed", "vpm_pct", "pom_pct", # structural
                  "stim_delay", "num_stims", "inter_stimulus_interval",  # stim. series
                  "vpm_mu", "pom_mu", "vpm_sigma", "pom_sigma", "vpm_spike_rate", "pom_spike_rate",
                  "vpm_proj_name", "pom_proj_name", "supersample", "data_for_vpm_input"]  # spikes

    cfg = {p: kwargs.get(p) for p in param_list}
    # Load circuit and fix targets
    c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])

    all_spike_times = []; all_spiking_gids = []
    fib_grps = ['vpm', 'pom']
    for fib_grp_i, fib_grp in enumerate(fib_grps):

        # gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], cfg["circuit_target"], cfg["supersample"])
        gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], "hex0", cfg["supersample"])
        centre_point = np.mean(pos, axis=0)


        new_gids = gids
        new_pos = pos

        # print(len(new_gids))

        # if (fib_grp == "vpm"):

        #     new_pos = []; new_gids = [];
        #     for p, gid in zip(pos, gids):
        #         # if (p[0] > centre_point[0]):

        #         euc_dist = np.linalg.norm(p - centre_point)
        #         print(euc_dist)
        #         # if (euc_dist < 125.0):

        #         # print(p, centre_point, np.linalg.norm(p - centre_point))

        #         new_pos.append(p)
        #         new_gids.append(gid)

            
        #     new_gids = np.asarray(new_gids)
        #     new_pos = np.asarray(new_pos)

        #     plt.figure()
        #     plt.scatter(pos[:, 0], pos[:, 1])
        #     plt.scatter(new_pos[:, 0], new_pos[:, 1])
        #     plt.gca().set_aspect('equal', adjustable='box')
        #     plt.savefig(os.path.join(path, fib_grp + '_stim_fibers.png'))
        #     plt.close()



            

        np.random.seed(cfg["stim_seed"] + fib_grp_i)

        pct_key = fib_grp + "_pct"

        if (cfg[pct_key] > 0.0):
            selected_gid_indices = np.sort(np.random.choice(range(len(new_gids)), size=int(len(new_gids) * cfg[pct_key]/100.), replace=False))
            selected_gids = new_gids[selected_gid_indices]
            selected_gid_poss = new_pos[selected_gid_indices]

            plt.figure()
            plt.scatter(pos[:, 0], pos[:, 1])
            plt.scatter(np.asarray(new_pos)[:, 0], np.asarray(new_pos)[:, 1])
            plt.scatter(np.asarray(selected_gid_poss)[:, 0], np.asarray(selected_gid_poss)[:, 1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(path, fib_grp + '_selected_stim_fibers.png'))
            plt.close()

            stim_times = spikewriter.generate_stim_series_num_stims(cfg["stim_delay"], cfg["num_stims"], cfg["inter_stimulus_interval"])

            # spike_times, spiking_gid_indices = spikewriter.generate_lognormal_spike_train(stim_times, selected_gid_indices,
            #                                         cfg[fib_grp + "_mu"], cfg[fib_grp + "_sigma"], cfg[fib_grp + "_spike_rate"], cfg["stim_seed"] + fib_grp_i)

            # spike_times, spiking_gid_indices = spikewriter.generate_yu_svoboda_spike_trains(stim_times, selected_gid_indices, cfg["data_for_vpm_input"], cfg["stim_seed"] + fib_grp_i)
            spike_times, spiking_gid_indices = spikewriter.generate_ji_diamond_estimate_scaled_spike_trains(stim_times, selected_gid_indices, cfg["stim_seed"] + fib_grp_i)


            spiking_gids = gids[spiking_gid_indices]


            plt.figure()
            plt.scatter(spike_times, spiking_gid_indices)
            plt.gca().set_xlim([1500, 1550])
            plt.savefig(os.path.join(path, fib_grp + 'stim_spikes.png'))
            plt.close()

            all_spike_times += spike_times.tolist()
            all_spiking_gids += spiking_gids.tolist()

        # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(all_spike_times, all_spiking_gids, os.path.join(path, stim_file))

    return {"stim_file": stim_file}


# Adapted from code created by Andras Ecker
def gen_whisker_flick_stim_and_find_fibers_centre(*, path, **kwargs):
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

        # gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], cfg["circuit_target"], cfg["supersample"])
        gids, pos = _get_projection_locations(path, c, cfg[fib_grp + "_proj_name"], "hex0", cfg["supersample"])
        centre_point = np.mean(pos, axis=0)

        if (fib_grp == "vpm"):

            new_pos = []; new_gids = [];
            for p, gid in zip(pos, gids):
                # if (p[0] > centre_point[0]):

                euc_dist = np.linalg.norm(p - centre_point)
                print(euc_dist)
                if (euc_dist < 125.0):

                # print(p, centre_point, np.linalg.norm(p - centre_point))

                    new_pos.append(p)
                    new_gids.append(gid)

            
            new_gids = np.asarray(new_gids)
            new_pos = np.asarray(new_pos)

            plt.figure()
            plt.scatter(pos[:, 0], pos[:, 1])
            plt.scatter(new_pos[:, 0], new_pos[:, 1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(path, fib_grp + '_stim_fibers.png'))
            plt.close()



            

        np.random.seed(cfg["stim_seed"] + fib_grp_i)

        pct_key = fib_grp + "_pct"

        if (cfg[pct_key] > 0.0):
            selected_gid_indices = np.sort(np.random.choice(range(len(new_gids)), size=int(len(new_gids) * cfg[pct_key]/100.), replace=False))
            selected_gids = new_gids[selected_gid_indices]
            selected_gid_poss = new_pos[selected_gid_indices]

            plt.figure()
            plt.scatter(pos[:, 0], pos[:, 1])
            plt.scatter(np.asarray(new_pos)[:, 0], np.asarray(new_pos)[:, 1])
            plt.scatter(np.asarray(selected_gid_poss)[:, 0], np.asarray(selected_gid_poss)[:, 1])
            plt.gca().set_aspect('equal', adjustable='box')
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


def generate_boundle_whisker_step_stim(*, path, **kwargs):
    """
    Generates whisker step (longer then flick) like VPM and optionally POm (on top) spike trains
    VPM fibers are clustered together (to e.g. scan toposample stim. params.) and as clustering happens on the fly,
    it's a more general version of `gen_whisker_flick_stim()` that can be applied to any version and region of the SSCx
    """
    param_list = ["circuit_config", "circuit_target", "user_target_path",  # base circuit
                  "stim_seed", "vpm_pct", "vpm_proj_name", "pom_pct", "pom_proj_name", "supersample", "n_clusters",  # structural
                  "reconn_delay", "sim_duration", "stim_delay", "stim_rate",  # stim. series
                  "stim_duration", "min_rate", "vpm_max_rate",  # VPM rate (POm rate is derived from these)
                  "bq", "tau"]  # spikes
    cfg = {p: kwargs.get(p) for p in param_list}
    # Load circuit and fix targets
    c = _load_circuit_targets(cfg["circuit_config"], cfg["user_target_path"])
    # VPM spike train
    # Get spatial location of fibers and cluster them to "bundles"
    gids, pos = _get_projection_locations(path, c, cfg["vpm_proj_name"], cfg["circuit_target"], cfg["supersample"])
    grp_gids, _, grp_idx = projloc.cluster_by_locations(gids, pos, n_clusters=cfg["n_clusters"])
    # Randomly select a given pct of bundles (preselect the ones with average size)
    len_grps = np.array([len(grp) for grp in grp_gids])
    m_lens, std_lens = np.mean(len_grps), np.std(len_grps)
    viable_grp_idx = np.where((len_grps > m_lens-std_lens) & (len_grps < m_lens+std_lens))[0]
    np.random.seed(cfg["stim_seed"])
    pattern_idx = np.random.choice(viable_grp_idx, size=int(cfg["n_clusters"] * cfg["vpm_pct"]/100.), replace=False)
    pattern_gids = np.sort(np.concatenate([gids[grp_idx == grp] for grp in pattern_idx]))
    # Generate the spike trains
    stim_times = spikewriter.generate_stim_series(cfg["stim_delay"], cfg["sim_duration"], cfg["stim_rate"])
    t, stim_rate = spikewriter.generate_rate_signal(cfg["reconn_delay"], cfg["sim_duration"], stim_times,
                                                    cfg["stim_duration"], cfg["min_rate"], cfg["vpm_max_rate"])
    spike_times, spiking_gids = spikewriter.generate_inh_adaptingmarkov_spike_train(pattern_gids, t, stim_rate,
                                            cfg["bq"], cfg["tau"], cfg["stim_seed"])
    # POm spike train (no clustering of fibers here)
    if cfg["pom_pct"] > 0.0:
        pom_max_rate = 0.5 * cfg["vpm_max_rate"]
        pom_gids, _ = _get_projection_locations(path, c, cfg["pom_proj_name"], cfg["circuit_target"], False)
        np.random.seed(cfg["stim_seed"])
        pom_gids = np.sort(np.random.choice(pom_gids, size=int(len(pom_gids) * cfg["pom_pct"]/100.), replace=False))
        _, pom_rate = spikewriter.generate_rate_signal(cfg["reconn_delay"], cfg["sim_duration"], stim_times,
                                                       cfg["stim_duration"], cfg["min_rate"], pom_max_rate)
        pom_spike_times, pom_spiking_gids = spikewriter.generate_inh_adaptingmarkov_spike_train(pom_gids, t, pom_rate,
                                                        cfg["bq"], cfg["tau"], cfg["stim_seed"])
        spike_times, spiking_gids = spikewriter.merge_spike_trains([spike_times, pom_spike_times],
                                                                   [spiking_gids ,pom_spiking_gids])
    # Write to file and return file name used in the template
    stim_file = "input.dat"
    spikewriter.write_spikes(spike_times, spiking_gids, os.path.join(path, stim_file))
    return {"stim_file": stim_file}


