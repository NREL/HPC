#!/bin/bash

# put the NREL modules in the module path
source /nopt/nrel/apps/env.sh

# clear all currently loaded modules
module purge

# load the modules necessary to set up the standalone Intel Compilers/Intel MPI environment
module load intel-oneapi-mpi
module load intel-oneapi
module unload cray-libsci
