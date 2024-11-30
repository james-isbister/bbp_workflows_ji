import numpy as np

stims_per_sim = 1
stim_gap = 2.0
num_sims = 72

string = "\"rotations\": ["


for sim_i in range(num_sims):
	string += "\""

	for stim_i in range(stims_per_sim):

		rotation = float(sim_i * stims_per_sim + stim_i*stim_gap)
		string += str(rotation)

		if stim_i < stims_per_sim - 1:
			string  += ", "

	string +="\""

	if sim_i < num_sims - 1:
		string += ",\n"

string += "]"
print(string)