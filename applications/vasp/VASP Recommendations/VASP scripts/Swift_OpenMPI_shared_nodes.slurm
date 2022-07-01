#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=parallel
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1

#Set --exclusive=user if you would like to prevent anyone else from running on the same nodes as you
#You will be charged for the full node regardless of the fraction of CPUs/node used
##SBATCH --exclusive=user

#To make sure that you are only being charged for the CPUs your job is using, set mem=2GB*CPUs/node
#--mem sets the memory used per node
#SBATCH --mem=46G

#Set how many jobs you would like to run at the same time as an array job
#In this example, an array of 2 jobs will be run at the same time. This script will be run once for each job.
#SBATCH --array=1-2

#The SLURM_ARRAY_TASK_ID variable can be used to modify the parameters of the distinct jobs in the array.
#In the case of array=1-2, the first job will have SLURM_ARRAY_TASK_ID=1, and the second will have SLURM_ARRAY_TASK_ID=2.
#For example, you could assign different input files to runs 1 and 2 by storing them in directories input_1 and input_2 and using the following code:

mkdir run_${SLURM_ARRAY_TASK_ID}
cd run_${SLURM_ARRAY_TASK_ID}
cp ../input_${SLURM_ARRAY_TASK_ID}/POSCAR .
cp ../input_${SLURM_ARRAY_TASK_ID}/POTCAR .
cp ../input_${SLURM_ARRAY_TASK_ID}/INCAR .
cp ../input_${SLURM_ARRAY_TASK_ID}/KPOINTS .

#Now load vasp and run the job...

module purge

#Load OpenMPI VASP build and necessary modules
ml vasp
ml slurm/21-08-1-1-o2xw5ti
ml openmpi/4.1.1-6vr2flz

#set cpu-bind in srun line
srun -n 64 vasp_std &> out
#srun --cpu-bind=cores -n 64 vasp_std &> out
#srun --cpu-bind=rank -n 64 vasp_std &> out

