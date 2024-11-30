cas = [1.05, 1.1]
depol_stdev_mean_ratios = [0.2, 0.4]
desired_connected_proportion_of_invivo_frss = [0.1, 0.3, 0.5, 0.7, 0.9]

count = -1
simulation_ids = []
for ca in cas:
	for depol_stdev_mean_ratio in depol_stdev_mean_ratios:
		for desired_connected_proportion_of_invivo_frs in desired_connected_proportion_of_invivo_frss:
			count += 1
			if ((ca in [1.05, 1.1]) & (depol_stdev_mean_ratio in [0.2, 0.4]) & (desired_connected_proportion_of_invivo_frs in [0.3, 0.7])):
				simulation_ids.append(count)

for sim_id in simulation_ids:
	print(str(sim_id) + ",", end = '')
# print(simulation_ids)