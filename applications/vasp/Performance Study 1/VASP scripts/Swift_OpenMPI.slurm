#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=test
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1

#Set --exclusive if you would like to prevent any other jobs from running on the same nodes (including your own)
#You will be charged for the full node regardless of the fraction of CPUs/node used
##SBATCH --exclusive

#To make sure that you are only being charged for the CPUs your job is using, set mem=2GB*CPUs/node
#--mem sets the memory used per node
#SBATCH --mem=128G

module purge

#Load OpenMPI VASP build and necessary modules
ml vasp 
ml slurm/21-08-1-1-o2xw5ti 
ml openmpi/4.1.1-6vr2flz

#set cpu-bind in srun line
srun -n 64 vasp_std &> out
#srun --cpu-bind=cores -n 64 vasp_std &> out
#srun --cpu-bind=rank -n 64 vasp_std &> out
