import os
import numpy as np
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


# get a sample of gids with the specified parameters > 50μm apart
def sample_presyn_targets(circuit, target, layer, morpho_class, sample=10):
    
    # approach - choose one from the specified group, and keep adding random samples 
    # from the same group until there is N cells that are a distance apart from each other
    # to scale up from 10 cells and 50um might require some advanced indexing/targets 
    
    init_sample=1000 #big enough sample to contain a subsample of pyramidal cells far enough from each other
    cell_group = circuit.cells.ids({'$target': target, Cell.LAYER: layer, Cell.MORPH_CLASS: morpho_class}, sample=init_sample)
    xyzs = circuit.cells.get(cell_group, properties=[Cell.X, Cell.Y, Cell.Z])
    
    N = sample-1
    sample_current = xyzs.sample(n=1)

    previously = []
    while len(previously) < N:
        #print("Found ", len(previously))
        sample_next = xyzs.sample(n=1)
    
        if sample_next.index[0] != sample_current.index[0]:
            previously.append(sample_current.index[0])
            if check_distance(sample_next.iloc[0], xyzs.loc[previously]):
                sample_current = sample_next

    sample_all = previously + [sample_current.index[0]]
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