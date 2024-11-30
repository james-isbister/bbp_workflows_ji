# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        J. Isbister, Adapted from C. Pokorny
# Date:          17/02/2021
# Last modified: 2/11/2021

import os
import shutil
import numpy as np
import pandas as pd
from bluepy import Circuit
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
    

def set_conductance_scalings_for_unconnected_frs(*, depol_stdev_mean_ratio, desired_connected_proportion_of_invivo_frs, in_vivo_reference_frs, unconnected_scaling_adjustment_denominators, data_for_unconnected_fit_name, **kwargs):

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
        

        data_for_unconnected_fit = pd.read_parquet(path=data_for_unconnected_fit_name)
        ng_data_for_plot = data_for_unconnected_fit[(data_for_unconnected_fit["ca"] == 1.15) & (data_for_unconnected_fit["neuron_class"] == scaling_and_data_neuron_class_keys[scaling_neuron_class_key])]
        # print(ng_data_for_plot)
        gradient_line_x = np.linspace(0,100,100000)
        gradient_line_y = gradient_line_x * depol_stdev_mean_ratio

        predicted_frs_for_line = griddata(ng_data_for_plot[['mean', 'stdev']].to_numpy(), ng_data_for_plot["data"], (gradient_line_x, gradient_line_y), method='cubic')

        in_vivo_fr = in_vivo_reference_frs[scaling_neuron_class_key]

        unconnected_scaling_adjustment_denominator = unconnected_scaling_adjustment_denominators[scaling_neuron_class_key]
        
        scaled_target_fr = desired_connected_proportion_of_invivo_frs*in_vivo_fr
        adjusted_unconnected_target_fr = scaled_target_fr/unconnected_scaling_adjustment_denominator

        # print(scaling_neuron_class_key, predicted_frs_for_line, adjusted_unconnected_target_fr)
        index_of_closest_point = np.nanargmin(abs(predicted_frs_for_line - adjusted_unconnected_target_fr))

        closest_x = round(gradient_line_x[index_of_closest_point], 3)
        closest_y = round(gradient_line_y[index_of_closest_point], 3)

        scale_dict.update({f'desired_connected_fr_{scaling_neuron_class_key}': scaled_target_fr})
        scale_dict.update({f'desired_unconnected_fr_{scaling_neuron_class_key}': adjusted_unconnected_target_fr})
        scale_dict.update({f'ornstein_uhlenbeck_mean_pct_{scaling_neuron_class_key}': closest_x})
        scale_dict.update({f'ornstein_uhlenbeck_sd_pct_{scaling_neuron_class_key}': closest_y})
        # scale_dict.update({f'shotn_mean_pct_{scaling_neuron_class_key}': closest_x})
        # scale_dict.update({f'shotn_sd_pct_{scaling_neuron_class_key}': closest_y})

        # print(scale_dict)

    return scale_dict

