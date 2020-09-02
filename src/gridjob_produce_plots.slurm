#!/bin/bash -x
#SBATCH --job-name=183_PLOT_ROUT
#SBATCH --account=covid19dynstat
#SBATCH --partition=batch
#SBATCH --array=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/p/project/covid19dynstat/autostart/runs/%A_o.txt
#SBATCH --error=/p/project/covid19dynstat/autostart/runs/%A_e.txt
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=t.kreuzer@fz-juelich.de
#SBATCH --dependency=afterok:3265

# select project
jutil env activate -p covid19dynstat

# ensure SLURM output/errors directory exists
# mkdir -p ${PROJECT}/${USER}/runs/

# run tasks
srun --exclusive -n ${SLURM_NTASKS} gridjob_produce_plots_183.slurm.sh
