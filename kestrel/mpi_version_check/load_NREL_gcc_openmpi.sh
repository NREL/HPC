#!/bin/bash

# put the NREL modules in the module path
source /nopt/nrel/apps/env.sh

# clear all currently loaded modules
module purge

# load the modules necessary to set up the standalone GNU GCC Compilers/OpenMPI environment
module load openmpi/4.1.5-gcc
module load gcc
