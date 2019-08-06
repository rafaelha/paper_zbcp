#!/bin/bash
#SBATCH --array=0-13
#SBATCH -p nodes # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 11000 # memory pool for all cores
#SBATCH -t 5-20:00 # time (D-HH:MM)
#SBATCH -o message.out # STDOUT
#SBATCH -e error.err # STDERR
export OMP_NUM_THREADS=1
module purge
module add anaconda3
python3 socconductance.py $SLURM_ARRAY_TASK_ID
