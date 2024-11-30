# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        J. Isbister, adapted from C. Pokorny, 
# Date:          15/12/2021
# Last modified: 16/12/2021

import os
import numpy as np
from bluepy import Circuit
import pick_postsyn_targets as picker
import generate_stimuli as gs
import matplotlib.pyplot as plt

""" Selects neurons for stimulation, finds their postsynapric targets and generates synchronous or rate coded spike trains. """
def rate_or_synchronous_stimulation_for_group_of_single_neurons(*, path, **kwargs):
    
    target_file_name = os.path.join(path, "postsynaptic_targets")
    circ = Circuit(kwargs['circuit_config'])
    circ_cfg_dict = circ.config.copy()
    for user_target_file in kwargs['custom_user_targets']:
        circ_cfg_dict['targets'].append(user_target_file) # Add user target file to list of existing target files
    circ = Circuit(circ_cfg_dict) # Re-load circuit

    stimulus_gids = picker.create_postsyn_targets(circ, 'hex0', kwargs['num_neurons_to_stimulate'], -1, target_file_name)    
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
    os.chown(user_target_file, uid=-1, gid=os.stat(path).st_gid)
    
    # print(f'INFO: Generated user target "{os.path.join(os.path.split(path)[-1], user_target_name)}"')
    
    return {'user_target_name': user_target_name}



def config_section_from_dict(sect_name, param_dict, intend=4):
    """
    Generates a BlueConfig section string from a dict
    """

    section_str = sect_name + '\n{\n'
    for k, v in param_dict.items():
        section_str += ' ' * intend + f'{k} {str(v)}\n' 
    section_str += '}\n\n'
    return section_str



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
        return np.round(scaled_value).astype(int)
    
    # Apply scaling
    shotn_scale_dict = {}
    for spec, scale in shotn_scale_factors.items(): # Specifier and scaling factor, e.g. "L1I": 0.6
        for tgt in shotn_scale_method.keys(): # Scaling target, e.g. "shotn_mean_pct"
            shotn_scale_dict.update({f'{tgt}_{spec}': scale_fct(kwargs[tgt], scale, shotn_scale_method[tgt])})    
    
    return shotn_scale_dict



