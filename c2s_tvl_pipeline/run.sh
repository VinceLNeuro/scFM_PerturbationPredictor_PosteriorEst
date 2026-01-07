#!/usr/bin/env bash

######## Slurm resource allocation ########
#SBATCH --job-name=c2s_tvl_11v3_fullLength_perturbation_PosteriorEst_Run3PosteriorEst
##SBATCH --cluster=htc
#SBATCH --time=5-00:00:00 #5:00:00
#SBATCH --nodes=1 #default - all cores on one machine
#SBATCH --ntasks-per-node=1 #default
#SBATCH --cpus-per-task=16 # number of cores (max)
#SBATCH --mem=768G         # total RAM  

#SBATCH --clusters=gpu  
#SBATCH --partition=a100_nvlink    #a100,a100_nvlink,l40s
#SBATCH --constraint=a100,80g,amd
#SBATCH --gres=gpu:1      # number of GPUs per node (gres=gpu:N)  

#SBATCH --account=hpark
#SBATCH --mail-user=til177@pitt.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --output=./%x_slurm%A.out        #./output/%x-slurm_%A.out

######## Load software into environment ########
module purge
source ~/conda_init.sh
conda activate cell2sentence

set -ev
# Confirm Python version and conda environment
echo "Using Python: $(which python)"
python -V
echo "Conda env: $CONDA_DEFAULT_ENV"

start_time=$(date +%s)
python c2s_tvl_11v3_fullLength_perturbation_PosteriorEst.py
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Duration: $duration seconds"

