import os
import numpy as np
import pandas as pd
from bluepy import Cell, Circuit

def save_target_to_file(gids_list, target_name, filename):

    #append 'a' if front of GIDs to be used as target
    gids=list(map(lambda x: 'a' + x, list(map(str,gids_list))))
    ' '.join(gids)
  
    with open(filename, 'a') as filehandle:
        filehandle.write('%s ' % 'Target Cell')
        filehandle.write('%s\n' % target_name)
        filehandle.write('%s\n' % '{')
        for gid in gids:
            filehandle.write('%s ' % gid)
        filehandle.write('\n%s' % '}\n')

# get a sample of interneurons postsynatpic to a presyn_gid
def get_postsyn_INTERNEURONS(circuit, presyn_gid, num_postsyn_gids):
    post_gids = circuit.connectome.efferent_gids(presyn_gid).astype(int)

    #Look up the morphology class of postsyn cells and select INTERNEURONS
    morph_cells = circuit.cells.get(group=post_gids, properties=[Cell.MORPH_CLASS])
    morph_class = np.where(morph_cells.values[:, 0] == 'INT', u'interneuron', morph_cells.values[:, 0])
    INTER_gids = morph_cells.index.values[morph_class=='interneuron']

    if (num_postsyn_gids == -1):
        return INTER_gids
    else:
        return np.random.choice(INTER_gids, np.minimum(num_postsyn_gids, len(INTER_gids)), replace=False)


def get_suitable_samples(cells, current_samples, min_dist=35):
    XYZ = ['x', 'y', 'z']
    dist_vec = cells[XYZ].to_numpy()[:,np.newaxis] - current_samples[XYZ].to_numpy()[np.newaxis,:]
    dist = np.linalg.norm(dist_vec, axis=2)
    mask = np.all(dist > min_dist, axis=1)
    return cells[mask]

# get a sample of L23 gids with the specified parameters at least 35μm apart
def sample_presyn_targets_L23(circuit, target, morpho_class, sample=10):
    from bluepy import Cell
    cell_group_2 = circuit.cells.ids({'$target': target, Cell.LAYER: 2, Cell.MORPH_CLASS: morpho_class})
    cell_group_3 = circuit.cells.ids({'$target': target, Cell.LAYER: 3, Cell.MORPH_CLASS: morpho_class})
    cell_group = np.concatenate((cell_group_2, cell_group_3), axis=None)
    
    xyzs = circuit.cells.get(cell_group, properties=[Cell.X, Cell.Y, Cell.Z])
    
    N = sample

    previously = pd.DataFrame(columns=xyzs.columns)
    suitable_samples = xyzs
    
    while len(previously) < N and not suitable_samples.empty: # Check if there are suitable samples left
        previously = pd.concat((previously, suitable_samples.sample(1)))
        suitable_samples = get_suitable_samples(suitable_samples, previously)
    
    sample_all = previously.index
    return sample_all



def create_postsyn_targets(circuit, target, num_presyn_gids, num_postsyn_gids, target_file):
    #remove target_file if exists
    try:
        os.remove(target_file)
    except OSError:
        pass
    
    PYR_gids = sample_presyn_targets(circuit, target, 3, "PYR", num_presyn_gids)
    print(len(PYR_gids))
    for PYR_gid in PYR_gids:
        post_gids=get_postsyn_INTERNEURONS(circuit, PYR_gid, num_postsyn_gids)
        print("selected post_gids for PYR_gid ", PYR_gid)
        #target_name="pyramidal_gid_" + str(PYR_gid)
        target_name=str(PYR_gid)
        save_target_to_file(post_gids, target_name, target_file)

    return PYR_gids


def check_distance(position, previous_locs, lower_bound=50):
    """..."""
    if not len(previous_locs):
        return True
    
    distances = np.linalg.norm(previous_locs - position, axis=1)
    return np.all(distances >= lower_bound)