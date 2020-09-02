#!/bin/bash -x
#SBATCH --job-name=RKI_IA
#SBATCH --account=covid19dynstat
#SBATCH --partition=batch
#SBATCH --array=100:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --output=/p/project/covid19dynstat/autostart/runs/%A_o.txt
#SBATCH --error=/p/project/covid19dynstat/autostart/runs/%A_e.txt
#SBATCH --time=24:00:00
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=<email>

# ATTENTION:
#     1. ensure SLURM output/errors directory exists
#     2. do not switch to zsh in ~/.bashrc because of "jutil env" 

# select project
jutil env activate -p covid19dynstat

# run tasks
srun --exclusive -n ${SLURM_NTASKS} gridjob_sample_ia.slurm.sh
