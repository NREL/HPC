#!/bin/bash

#SBATCH --ntasks=1 # Tasks to be run
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=60   # Required, estimate 5 minutes
#SBATCH --account= <account_name_here> # Required Talk to Wes about what you should use
#SBATCH -o output.txt
#SBATCH -e errors.txt
#SBATCH --partition=debug
#SBATCH --gres=gpu:1


cd /scratch/pdiaz/
module use /nopt/nrel/apps/modules/centos74/modulefiles/
module load gcc/7.4.0
module load cuda/10.0.130
module load cudnn/7.4.2/cuda-10.0
module load conda
sleep 3
source activate py38tf23
sleep 5
python3 TFbenchmark.py
