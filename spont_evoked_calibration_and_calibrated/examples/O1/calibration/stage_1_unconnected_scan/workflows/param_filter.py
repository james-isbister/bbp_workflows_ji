# Description:   BBP-WORKFLOW parameter processor functions used to generate SSCx simulation campaigns
# Author:        J. Isbister
# Last modified: 2022-2023


def depol_stdev_mean_ratio_filter(*, 
	                    depol_mean, 
	                    depol_std, 
	                    min_depol_stdev_mean_ratio,
	                    max_depol_stdev_mean_ratio,
	                    **kwargs):
	
	sim_ratio = depol_std / depol_mean

	if (sim_ratio >= min_depol_stdev_mean_ratio) & (sim_ratio <= max_depol_stdev_mean_ratio): 
		return True

	return False
