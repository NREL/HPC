#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00-00:30:00
#SBATCH --ntasks-per-node=4

# RUN REQUIREMENTS: You must do the following to actual use this script
#    - add the line
#           #SBATCH --account=<project_handle>
#      with an appropriate resource allocation in place of <project_handle>
#    - create a conda environment named 'py-jl-mpi' with the following
#      python modules installed
#        * python=3
#        * numpy
#        * mpi4py (installed against the same mpi module that is loaded below)
#        * julia
#    - an install of julia with the environment given by ./Project.toml
#      (relative to this script) instantiated and PyCall installed in
#      the default julia environment
# See the README.md in this directory for instructions on setting this up.

module purge

module load conda/2019.10
## Setup shell to use conda activate
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nopt/nrel/apps/anaconda/2019.10/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nopt/nrel/apps/anaconda/2019.10/etc/profile.d/conda.sh" ]; then
        . "/nopt/nrel/apps/anaconda/2019.10/etc/profile.d/conda.sh"
    else
        export PATH="/nopt/nrel/apps/anaconda/2019.10/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

module load openmpi/3.1.6/gcc-8.4.0

conda activate py-jl-mpi

srun python ./mpi_jl_hello_world.py
srun python ./mpi_jl_pi.py
srun python ./mpi_jl_pi_as_lib.py
srun python ./mpi_jl_cv_pi.py
