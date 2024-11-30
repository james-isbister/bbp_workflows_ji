# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        J. Isbister, Adapted from C. Pokorny
# Date:          2/11/2022

import os
import shutil
import numpy as np
import pandas as pd
from bluepy import Circuit
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import math
from scipy.stats import linregress


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

    # quick switch so it runs. Already have data for these conditions
    if (ca == 1.1):
        if (depol_stdev_mean_ratio in [0.2, 0.4]):
            ca = 1.05


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
        ca_unconn_conn_df = unconn_conn_df[(unconn_conn_df['ca']==ca) & (unconn_conn_df['depol_stdev_mean_ratio']==depol_stdev_mean_ratio) &  (unconn_conn_df["window"] == 'conn_spont') & (unconn_conn_df['bursting'] == False)]
        # ca_unconn_conn_df = unconn_conn_df[(unconn_conn_df["window"] == 'conn_spont') & (unconn_conn_df['bursting'] == False)]

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

                desired_unconnected_fr = nc_ca_unconn_conn_df['desired_connected_fr'] / ca_lr_slope

            print(ca, neuron_class, desired_connected_fr, desired_unconnected_fr)
        
        else: 
            desired_unconnected_fr = desired_connected_fr



        nc_unconn_df = unconn_df[(unconn_df["ca"] == 1.15) & (unconn_df["neuron_class"] == scaling_and_data_neuron_class_keys[scaling_neuron_class_key])]
        
        # print(ng_data_for_plot)
        gradient_line_x = np.linspace(0,100,100000)
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
