#!/bin/bash
# available SLURM env.vars: https://slurm.schedmd.com/sbatch.html#lbAK

# create task output directory
TASK_ID=$((${SLURM_ARRAY_TASK_ID}+${SLURM_LOCALID}))
TASK_DIR=${SCRATCH}/run_${SLURM_ARRAY_JOB_ID}/task_${TASK_ID}
mkdir -p ${TASK_DIR}
echo "TASK ${TASK_ID}: Running in job-array ${SLURM_ARRAY_JOB_ID} on `hostname` and dump output to ${TASK_DIR}"

# activate virtual python environment
source ${PROJECT}/.local/share/venvs/covid19dynstat_v01/bin/activate

# run code
export SGE_TASK_ID=${TASK_ID} # needed for later python scripts
THEANO_FLAGS="base_compiledir=${TASK_DIR}/,floatX=float32,device=cpu,openmp=True,mode=FAST_RUN,warn_float64=warn" OMP_NUM_THREADS=8  python3 sample_ia_effects.py > ${TASK_DIR}/log.txt
