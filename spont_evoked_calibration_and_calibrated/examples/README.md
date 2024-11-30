# Calibration & calibrated cortical circuits (SONATA)
This directory contains workflows for simulating calibrated cortical circuits and running calibrations with SONATA.

For details on the calibrated simulations & calibration approach see:
[Modeling and Simulation of Neocortical Micro- and Mesocircuitry. Part II: Physiology and Experimentation. Isbister et al., 2023. bioRxiv.](www.biorxiv.org/content/10.1101/2023.05.17.541168v3)

After creating a [BBP Workflow](https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Workflow) virtual environment. Please install the requirements.txt

## _Calibration / Calibrated_
- **Calibration**: The calibration procedure is divided into three major stages represented by 3 corresponding workflows (see `O1/calibration` for further details). In many cases, the 1st and 2nd stages can be avoided if data under similar conditions exists.
- **Calibrated** simulations have had conductance injections to different populations / subpopulations optimised so that the firing rates of different populations / subpopulations match target in vivo firing rates multiplied by a specified proportion P\_FR. Calibrations can be made for different values of extraceullular calcium concentration (Ca^2+) and the ratio between the standard deviation and mean of the injected conductances across populations (R\_OU). 


## _Circuits_
Workflows are provided for the following circuits / subvolumnes, although the calibration workflows should be general to other cortical circuits.
- **nbS1**: The non-barrel somatasensory cortex (rat).
- **O1 (nbS1)**: A subvolume of the nbS1 model corresponding to 7 hexagonal columns.

## _Populations_
Calibration of conductance injection is made differentially at two levels of granularity:
- **EI populations** (9: L1I, L2/3E, L2/3I, L4E, L4I, L5E, L5I, L6E, L6I)
- **EI subpopulations**: (17: L1\_5HT3aR, L23E, L23PV, L23SST, L23\_5HT3aR, L4E, ...)

<!-- ## _Stimuli_
- **Thalamic stimuli** (e.g. Stimulating proportion of thalamic fibers)
- **Spike replay**
- **Optogenetic stimulation**

## _Extracellular / LFP_
- **Extracellular potentials** -->