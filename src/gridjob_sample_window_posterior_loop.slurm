#!/bin/bash -x
#SBATCH --job-name=POST_ROUT
#SBATCH --account=covid19dynstat
#SBATCH --partition=batch
#SBATCH --array=1-150
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/p/project/covid19dynstat/autostart/runs/%A_o.txt
#SBATCH --error=/p/project/covid19dynstat/autostart/runs/%A_e.txt
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.kreuzer@fz-juelich.de
# select project
jutil env activate -p covid19dynstat

# ensure SLURM output/errors directory exists
# mkdir -p ${PROJECT}/${USER}/runs/

# run tasks
srun --exclusive -n ${SLURM_NTASKS} gridjob_sample_window_posterior_loop.slurm.sh
