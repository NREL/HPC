#!/bin/tcsh
#SBATCH --job-name n1.1
#SBATCH --ntasks=36
#SBATCH --time=1:00:00
#SBATCH --account=hpcapps
#SBATCH --error=std.err
#SBATCH --output=std.out
#SBATCH -p debug
#SBATCH --mem=184000 

module purge
module use /nopt/nrel/apps/modules/centos77/modulefiles/
module load intel-mpi lammps/20200303 mkl
module list

cp ../input/* .
bunzip2 *.bz2

time srun -n 36 lmp -in large.in >& md.log


