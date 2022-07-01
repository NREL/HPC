#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --nodes=1

module purge

#Load Intel MPI VASP build
ml vasp openmpi/4.0.4/gcc-8.4.0

#set cpu-bind in srun line
srun -n 36 vasp_std &> out
#srun --cpu-bind=cores -n 36 vasp_std &> out
#srun --cpu-bind=rank -n 36 vasp_std &> out
