# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        J. Isbister
# Last modified: 2022-2023

import numpy as np
import pandas as pd
import hashlib
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.stats import linregress
import math

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


# Used for fitting exponential
# between unconn and conn FRs
# i.e. phi: U_FR -> C_FR
def fit_exponential(x, y):

    popt, pcov = curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c,
        x, y, p0=(1.0, 0.5, y.min() - 1), 
        maxfev=20000)
    return popt


# Used for applying 
# phi inverse: C_FR -> U_FR
def calculate_suggested_unconnected_firing_rate(target_connected_fr, 
                                                            a, b, c):
    log_domain = max((target_connected_fr - c) / a, 1.0)
    return math.log(log_domain) / b



def set_input_compensation_for_desired_frs(*, 
                    path, 
                    ca, 
                    depol_stdev_mean_ratio, 
                    desired_connected_proportion_of_invivo_frs, 
                    data_for_unconnected_fit_name, 
                    data_for_connected_adjustment_fit_name, 
                    in_vivo_reference_frs,
                    unconnected_connected_fr_adjustment_fit_method='exponential', 
                    **kwargs):

    parameters = {}

    ### 1. Calculate phi[R_OU, Ca2+]: U_FR -> C_FR, 
    ### for simulation R_OU, Ca2+ and apply phi inverse

    if (data_for_connected_adjustment_fit_name == ''):
        # If no connected firing rate data, then this is the first optimisation
        # iteration, and the unconnected firing rates equal the target 
        # connected firing rates.
        desired_unconnected_fr = desired_connected_fr
    else:
        # Otherwise, load the connected firing rate data
        unconn_conn_df = pd.read_parquet(path=data_for_connected_adjustment_fit_name)
        
        # Filter non bursting sims from firing rate data and filter for the current simulation Ca2+ and R_OU
        ca_unconn_conn_df = unconn_conn_df[(unconn_conn_df['ca']==ca) 
                            & (unconn_conn_df['depol_stdev_mean_ratio']==depol_stdev_mean_ratio) 
                            & (unconn_conn_df["window"] == 'conn_spont') 
                            & (unconn_conn_df['bursting'] == False)]

        # Preprocess data
        ca_unconn_conn_df["invivo_fr"] = ca_unconn_conn_df['desired_connected_fr'] / ca_unconn_conn_df['desired_connected_proportion_of_invivo_frs']
        ca_unconn_conn_df = ca_unconn_conn_df[ca_unconn_conn_df['mean_of_mean_firing_rates_per_second'] < ca_unconn_conn_df['invivo_fr'] * 1.05]
    
    # Iterate neuron classes
    for scaling_neuron_class_key, in_vivo_fr in in_vivo_reference_frs.items():

        neuron_class = scaling_and_data_neuron_class_keys[scaling_neuron_class_key]
        in_vivo_fr = in_vivo_reference_frs[scaling_neuron_class_key]
        desired_connected_fr = desired_connected_proportion_of_invivo_frs*in_vivo_fr
        nc_ca_unconn_conn_df = ca_unconn_conn_df[ca_unconn_conn_df['neuron_class']==neuron_class]

        # Calculate phi and apply phi inverse
        if (unconnected_connected_fr_adjustment_fit_method == 'exponential'):            

            popt = fit_exponential(nc_ca_unconn_conn_df['desired_unconnected_fr'], nc_ca_unconn_conn_df['mean_of_mean_firing_rates_per_second'])
            desired_unconnected_fr = calculate_suggested_unconnected_firing_rate(desired_connected_fr, popt[0], popt[1], popt[2])

        elif (unconnected_connected_fr_adjustment_fit_method == 'linear'):

            ca_lr = linregress(ca_stat1, ca_stat2)
            ca_lr_slope = np.around(ca_lr.slope, 3)

            desired_unconnected_fr = nc_ca_unconn_conn_df['desired_unconnected_fr'] / ca_lr_slope
            


        ### 2. Calculate xi: U_FR -> OU_mu for specific R_OU value, and apply
        unconn_df = pd.read_parquet(path=data_for_unconnected_fit_name)
        nc_unconn_df = unconn_df[unconn_df["neuron_class"] == scaling_and_data_neuron_class_keys[scaling_neuron_class_key]]
        gradient_line_x = np.linspace(0,100,1000)
        gradient_line_y = gradient_line_x * depol_stdev_mean_ratio
        predicted_frs_for_line = griddata(nc_unconn_df[['mean', 'stdev']].to_numpy(), nc_unconn_df["data"], (gradient_line_x, gradient_line_y), method='cubic')
        index_of_closest_point = np.nanargmin(abs(predicted_frs_for_line - desired_unconnected_fr))
        closest_x = round(gradient_line_x[index_of_closest_point], 3)
        closest_y = round(gradient_line_y[index_of_closest_point], 3)


        # 3. Fill return dictionary
        parameters.update({f'desired_connected_fr_{scaling_neuron_class_key}': desired_connected_fr})
        parameters.update({f'desired_unconnected_fr_{scaling_neuron_class_key}': desired_unconnected_fr})
        parameters.update({f'ornstein_uhlenbeck_mean_pct_{scaling_neuron_class_key}': closest_x})
        parameters.update({f'ornstein_uhlenbeck_sd_pct_{scaling_neuron_class_key}': closest_y})

        print(parameters)

    return parameters
