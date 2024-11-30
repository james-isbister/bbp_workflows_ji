cas = [1.05, 1.1]
depol_stdev_mean_ratios = [0.3, 0.4]
vpm_pcts = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
desired_connected_proportion_of_invivo_frss = [0.1, 0.3, 0.5]
# vpm_l5e_cond_scaling_factors = [1.0, 1.36]

count = -1
simulation_ids = []
for ca in cas:
	for depol_stdev_mean_ratio in depol_stdev_mean_ratios:
		for vpm_pct in vpm_pcts:
			for desired_connected_proportion_of_invivo_frs in desired_connected_proportion_of_invivo_frss:
				# for vpm_l5e_cond_scaling_factor in vpm_l5e_cond_scaling_factors:
				count += 1
				# if ((ca in [1.05, 1.1]) & (depol_stdev_mean_ratio in [0.3, 0.4]) & (vpm_pct in [5.0, 10.0, 15.0]) & (desired_connected_proportion_of_invivo_frs in [0.1, 0.3, 0.5]) & (vpm_l5e_cond_scaling_factor in [1.36])):
				# if ((ca in [1.05, 1.1]) & (depol_stdev_mean_ratio in [0.2, 0.3, 0.4]) & (vpm_pct in [5.0]) & (desired_connected_proportion_of_invivo_frs in [0.1, 0.3, 0.5, 0.7]) & (vpm_l5e_cond_scaling_factor in [1.36])):
				# if ((ca in [1.05]) & (depol_stdev_mean_ratio in [0.3, 0.4]) & (vpm_pct in [5.0, 10.0, 15.0]) & (desired_connected_proportion_of_invivo_frs in [0.1, 0.3, 0.5]) & (vpm_l5e_cond_scaling_factor in [1.0])):
				if ((ca in [1.05]) & (depol_stdev_mean_ratio in [0.3]) & (vpm_pct in [5.0, 10.0, 15.0]) & (desired_connected_proportion_of_invivo_frs in [0.1, 0.3, 0.5])):
					simulation_ids.append(count)
				elif ((ca in [1.1]) & (depol_stdev_mean_ratio in [0.3, 0.4]) & (vpm_pct in [5.0, 10.0]) & (desired_connected_proportion_of_invivo_frs in [0.1, 0.3, 0.5])):
					simulation_ids.append(count)


for sim_id in simulation_ids:
	print(str(sim_id) + ",", end = '')
# print(simulation_ids)