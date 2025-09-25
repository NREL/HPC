#!/bin/bash
#SBATCH --account=$HPC_HANDLE
#SBATCH --time=4:00:00
#SBATCH --job-name=rl_train
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --tasks-per-node=1

# Step 1. Loading conda env
module purge
module load anaconda3
conda activate /projects/$HPC_HANDLE/$USER/conda_envs/rl_hpc

# Step 2. Starting Ray cluster.
# This step is skipped for the single node training.

# Step 3. Starting RL training
python -u train_script.py --run PPO
