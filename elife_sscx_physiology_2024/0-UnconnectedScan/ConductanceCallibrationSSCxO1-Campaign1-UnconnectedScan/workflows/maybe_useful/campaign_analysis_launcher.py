# Description: BBP-WORKFLOW code for launching arbitrary simulation campaign analyses
# Author: C. Pokorny
# Created: 28/06/2021

import os
import shutil
import json
import numpy as np
import subprocess
from collections.abc import Mapping
from luigi import Task, Parameter, ListParameter, DictParameter
from bbp_workflow.simulation import LookupSimulationCampaign
from bbp_workflow.utils import xr_from_dict
from bbp_workflow.luigi import RunAnywayTarget
from bbp_workflow.task import SbatchTask

DEFAULT_ARCHIVE = 'unstable'
DEFAULT_MODULES = 'python py-bluepy'

""" Helper function to recursively unfreeze and convert Luigi's FrozenOrderedDict parameter objects to regular dicts """
def unfreeze_recursively(value):
    if isinstance(value, Mapping):
        return {k: unfreeze_recursively(v) for k, v in value.items()}
    elif isinstance(value, list) or isinstance(value, tuple):
        return [unfreeze_recursively(v) for v in value]
    return value


""" Campaign analysis launcher, preparing config files and launching separate analysis tasks as specified """
class CampaignAnalysisLauncher(Task):
    
    list_of_analyses = ListParameter(default=[])
    
    def requires(self):
        return LookupSimulationCampaign()
    
    def run(self):
        
        # Load simulation campaign config from Nexus URL
        sim_campaign_cfg = sim_campaign_cfg = self.input().entity
        config = xr_from_dict(sim_campaign_cfg.configuration.as_dict())  # Get sim campaign config as Xarray
        root_path = os.path.join(config.attrs['path_prefix'], config.name) # Root path of simulation campaign
        sim_paths = config.to_series() # Single simulation paths as Pandas series with multi-index
        if not os.path.isabs(os.path.commonpath(sim_paths.tolist())): # Create absolute paths, if necessary
            for idx in range(sim_paths.shape[0]):
                sim_paths.iloc[idx] = os.path.join(os.path.split(root_path)[0], sim_paths.iloc[idx])
        assert os.path.commonpath(sim_paths.tolist() + [root_path]) == root_path, 'ERROR: Root path mismatch!'
        
        print(f'\nINFO: Loaded simulation campaign "{sim_campaign_cfg.name}" from {sim_campaign_cfg.get_url()} with coordinates {list(sim_paths.index.names)}')
        
        # Check if simulation results exist
        valid_sims = [os.path.exists(os.path.join(p, 'out.dat')) for p in sim_paths]
        sim_paths = sim_paths[valid_sims]
        
        print(f'INFO: Found {np.sum(valid_sims)} of {len(valid_sims)} completed simulations to analyze')
                
        # Create simulation paths to BlueConfigs
        sims = sim_paths.apply(lambda p: os.path.join(p, 'BlueConfig'))
        assert np.all([os.path.exists(s) for s in sims.values.flatten()]), 'ERROR: BlueConfig(s) missing!'
        
        # Prepare & launch analyses, as specified in launch config
        num_analyses = len(self.list_of_analyses)
        print(f'INFO: {num_analyses} campaign {"analysis" if num_analyses == 1 else "analyses"} to launch: {[anlys["name"] for anlys in self.list_of_analyses]}')
        launch_path = os.path.join(root_path, 'analyses')
        if not os.path.exists(launch_path):
            os.makedirs(launch_path)

        # Write (unfiltered; but only existing) simulation file
        sim_file = 'simulations.pkl'
        sims.to_pickle(os.path.join(launch_path, sim_file))

        analysis_tasks = []
        for anlys in self.list_of_analyses:
            anlys_name = anlys['name']
            anlys_repo = anlys['repository']
            anlys_checkout = anlys['checkout_id']
            anlys_script = anlys['script']
            anlys_params = unfreeze_recursively(anlys['parameters'])
            anlys_res = anlys['resources']
            
            # Create script and output folders
            script_path = os.path.join(launch_path, 'scripts', anlys_name)
            if not os.path.exists(script_path):
                os.makedirs(script_path)
            output_path = os.path.join(launch_path, 'output', anlys_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            # Apply conditions filter to apply the analysis only on a subset of simulation conditions (optional)
            cond_filter = anlys.get("condition_filter", {})
            sims_filt = sims.copy()
            for k, v in cond_filter.items():
                assert k in sims_filt.index.names, f'ERROR: Condition {k} not found for filtering in "{anlys_name}" analysis! Conditions: {sims_filt.index.names}'
                sims_filt = sims_filt[np.isin(sims_filt.index.get_level_values(k), v)]
            assert sims_filt.size > 0, f'ERROR: No simulations left for "{anlys_name}" analysis after condition filtering!'
            if sims_filt.size < sims.size:
                print(f'INFO: Selected {sims_filt.size} of {sims.size} simulations for "{anlys_name}" analysis')
            
            # Write simulation file
            sims_filt.to_pickle(os.path.join(script_path, sim_file))
            
            # Write analysis parameters
            anlys_params['output_root'] = output_path
            param_file = 'parameters.json'
            with open(os.path.join(script_path, param_file), 'w') as f:
                json.dump(anlys_params, f, indent=2)
            
            # Clone GIT repository to script_path, using branch/tag/hash as specified
            # [WORKAROUND: Needs to be launched on BB5, so that git is available]
            # TODO: Create/use venv in case of specific dependencies that are not part of the repository or a module
            repo_name = os.path.splitext(os.path.split(anlys_repo)[-1])[0]
            repo_path = os.path.join(script_path, repo_name)
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path, ignore_errors=True) # Remove if already exists
            proc = subprocess.Popen(f'cd {script_path};\
                                      git clone --no-checkout {anlys_repo};\
                                      cd {repo_name};\
                                      git checkout {anlys_checkout}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(proc.communicate()[0].decode())
            script_file = os.path.join(repo_name, anlys_script) # Path relative to script_path
            assert os.path.isfile(os.path.join(script_path, script_file)), 'ERROR: Script file not found!'
            
            # Prepare tasks
            cmd = f'python -u {script_file}'
            args = f'{sim_file} {param_file}'
            module_archive = anlys.get('module_archive', DEFAULT_ARCHIVE)
            modules = (DEFAULT_MODULES + ' ' + anlys.get('modules', '')).strip()
            anlys_res = {k: str(v) for k, v in anlys_res.items()} # Convert values to str, to avoid warning from parameter parser when directly passing whole "resources" dict
            analysis_tasks.append(CampaignAnalysis(name=anlys_name, chdir=script_path, command=cmd, args=args, module_archive=module_archive, modules=modules, **anlys_res))
        
        yield analysis_tasks # Launch tasks
        
        self.output().done()
    
    def output(self):
        return RunAnywayTarget(self)


""" Campaign analysis task, running an analysis as SLURM job """
class CampaignAnalysis(SbatchTask):
    
    name = Parameter()
    
    def run(self):
        print(f'\nINFO: Running campaign analysis task "{self.name}"\n')

        self.job_name = 'CampaignAnalysis[' + self.name + ']'
        super().run()
        
        self.output().done()
    
    def output(self):
        return RunAnywayTarget(self)
