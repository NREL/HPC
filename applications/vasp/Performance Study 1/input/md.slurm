#!/bin/tcsh
#SBATCH --job-name v.test
#SBATCH --ntasks=36
#SBATCH --time=1:00:00
#SBATCH --account=hpcapps
#SBATCH --error=std.err
#SBATCH --output=std.out
#SBATCH -p debug

module purge
module use /nopt/nrel/apps/modules/centos77/modulefiles/
module load vasp/6.1.1 intel-mpi/2020.1.217 mkl/2020.1.217 
module list

time srun -n 36 vasp_std >& md.log


