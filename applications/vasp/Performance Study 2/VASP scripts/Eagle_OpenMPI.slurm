#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --nodes=1

module purge

#Load Open MPI VASP build
source /nopt/nrel/apps/210830a/myenv.2108301742
ml vasp/6.1.1-l2mkbb2

#set cpu-bind in the srun line
srun -n 36 vasp_std &> out
#srun --cpu-bind=cores -n 36 vasp_std &> out
#srun --cpu-bind=rank -n 36 vasp_std &> out
