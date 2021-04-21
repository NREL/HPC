#!/bin/bash

#SBATCH --ntasks=1 # Tasks to be run
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=60   # Required, estimate 5 minutes
#SBATCH --account=<update_this_field> # Required Talk to Wes about what you should use
#SBATCH -o output.txt
#SBATCH -e errors.txt
#SBATCH --partition=debug


cd /scratch/$USER/tensorflow/
module use /nopt/nrel/apps/modules/centos74/modulefiles/
module load conda
module load gcc/7.4.0
module load cuda/10.0.130
module load cudnn/7.4.2/cuda-10.0
sleep 3
source activate py38tf23
sleep 5
export TMP=/scratch/$USER/bazel_build
export TEST_TMPDIR=/scratch/$USER/bazel_build
export TMPDIR=/scratch/$USER/bazel_build
unset LD_PRELOAD
bazel --output_base=/scratch/$USER/bazel_build build -c opt --copt=-O3 --copt=-march=skylake-avx512 --copt=-mtune=skylake-avx512 --copt=-Wno-sign-compare --copt=-Wno-unused  -k //tensorflow/tools/pip_package:build_pip_package
