# Description: Example analysis to be used with BBP-WORKFLOW analysis launcher
# Author: C. Pokorny
# Created: 29/06/2021

import sys
import json
import os
import pandas as pd
from bluepy import Simulation
import matplotlib.pyplot as plt

def main():
    
    # Parse inputs
    args = sys.argv[1:]
    if len(args) < 2:
        print(f'Usage: {__file__} simulations.pkl config_file.json')
        sys.exit(2)
    
    # Load simulation table
    sims = pd.read_pickle(args[0])
    
    # Load analysis parameters
    with open(args[1], 'r') as f:
        params = json.load(f)
    
    # Get params
    output_root = params.get('output_root')
    assert not output_root is None, 'ERROR: Output root folder not specified!'
    groupby = params.get('groupby')
    
    # Run analysis (= simple spike raster plotting)
    cond_names = sims.index.names
    for cond, path in sims.iteritems():
        cond_dict = dict(zip(cond_names, cond))
        
        sim = Simulation(path)
        t_start = params.get('t_start', sim.t_start)
        t_end = params.get('t_end', sim.t_end)
        
        fig = plt.figure()
        sim.plot.raster(sample=None, groupby=groupby, t_start=t_start, t_end=t_end)
        plt.xlim((t_start, t_end))
        plt.title(cond_dict)
        if not groupby is None:
            plt.legend(title=groupby, loc='upper right')
        file_name = 'spikes__' + '__'.join([f'{k}_{v}' for k, v in cond_dict.items()]) + '.png'
        fig.savefig(os.path.join(output_root, file_name), dpi=300)


if __name__ == "__main__":
    main()
