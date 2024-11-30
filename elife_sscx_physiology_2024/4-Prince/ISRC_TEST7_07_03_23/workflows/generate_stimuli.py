import sys
from elephant import spike_train_generation as e_stg
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import os


def write_clip_states(clip_states, clip_start_and_end_times, fn):
	if not os.path.exists(os.path.split(fn)[0]):
		os.makedirs(os.path.split(fn)[0])
	with open(fn, 'w') as fl:
		for s, s_e in zip(clip_states, clip_start_and_end_times):
			fl.write('%d\t%f\t%f\n' % (s, s_e[0], s_e[1]))

def generate_1_bit_pattern(experiment, fn, pulse_window):

	np.random.seed(1)

	total = experiment['stimulation_window'][0]
	clip_state = True
	clip_start_and_end_times = []
	clip_states = []
	while (total < experiment['stimulation_window'][1]):
		
		clip_start_time = total

		clip_length = np.ceil(np.random.uniform(low=experiment['clip_width_range'][0] / pulse_window, high=experiment['clip_width_range'][1] / pulse_window, size=1)[0]) * pulse_window
		total += clip_length
		clip_end_time = total

		clip_start_and_end_times.append([clip_start_time, clip_end_time])
		clip_states.append(clip_state)

		# Change state
		clip_state = not clip_state

	write_clip_states(clip_states, clip_start_and_end_times, fn)

	return clip_states, clip_start_and_end_times





def write_spikes(spike_times, spike_ids, fn):
	if not os.path.exists(os.path.split(fn)[0]):
		os.makedirs(os.path.split(fn)[0])
	with open(fn, 'w') as fl:
		fl.write('/scatter\n')
		for t, i in zip(spike_times, spike_ids):
			fl.write('%f\t%d\n' % (t, i))

def generate_spikes_from_1_bit_pattern(experiment, clip_states, clip_start_and_end_times, stimulus_gids, pulse_window):

	spike_times_by_i = [[] for gid in stimulus_gids]
	spike_ids_by_i = [[] for gid in stimulus_gids]
	for clip_state, clip_start_and_end_time in zip(clip_states, clip_start_and_end_times):

		clip_state_type = "Asynchronous"
		if ((experiment['experiment_type'] == "SynchronousExperiment") & (clip_state)):
			clip_state_type = "Synchronous"

		if (clip_state_type == "Synchronous"):
			spikes = (np.floor(e_stg.homogeneous_poisson_process((experiment['up_and_down_rates'][int(clip_state)]*pulse_window)*pq.Hz, t_start=(clip_start_and_end_time[0]/pulse_window)*pq.ms, t_stop=(clip_start_and_end_time[1]/pulse_window)*pq.ms, as_array=True))*pulse_window)
			spikes += np.random.normal(loc=5.0, scale=1.0, size=spikes.size)
			spikes = spikes.tolist()

		for i, stimulus_gid in enumerate(stimulus_gids):

			if (clip_state_type == "Asynchronous"):
				spikes = (np.floor(e_stg.homogeneous_poisson_process((experiment['up_and_down_rates'][int(clip_state)]*pulse_window)*pq.Hz, t_start=(clip_start_and_end_time[0]/pulse_window)*pq.ms, t_stop=(clip_start_and_end_time[1]/pulse_window)*pq.ms, as_array=True))*pulse_window)
				spikes += np.random.normal(loc=5.0, scale=1.0, size=spikes.size)
				spikes = spikes.tolist()

			spike_times_by_i[i].extend(spikes)
			spike_ids_by_i[i].extend([stimulus_gid for s in spikes])

	return spike_times_by_i, spike_ids_by_i

