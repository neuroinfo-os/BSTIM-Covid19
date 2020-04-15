#!/bin/bash -x
#SBATCH --job-name=RKI_IA
#SBATCH --account=covid19dynstat
#SBATCH --partition=batch
#SBATCH --array=1-100:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --output=/p/project/covid19dynstat/%u/runs/%A_o.txt
#SBATCH --error=/p/project/covid19dynstat/%u/runs/%A_e.txt
#SBATCH --time=24:00:00
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=<email>

# select project
jutil env activate -p covid19dynstat

# ensure SLURM output/errors directory exists
mkdir -p ${PROJECT}/${USER}/runs/

# run tasks
srun --exclusive -n ${SLURM_NTASKS} gridjob_sample_ia.slurm.sh
