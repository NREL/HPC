#!/bin/bash

#SBATCH --ntasks=1 # Tasks to be run
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=60   # Required, estimate 5 minutes
#SBATCH --account= <account_name_here> # Required Talk to Wes about what you should use
#SBATCH -o output.txt
#SBATCH -e errors.txt
#SBATCH --partition=debug
#SBATCH --gres=gpu:1


cd /scratch/$USER/
# the modules loaded below need to match the versions described in README.md for
# the corresponding TensorFlow version
module purge
module use /nopt/nrel/apps/modules/test/modulefiles/
module load conda
module load gcc/7.4.0
module load cudnn/8.0.5/cuda-10.2
sleep 3
source activate py38tf24
sleep 5
python3 TFbenchmark.py
