#!/bin/tcsh
#SBATCH --job-name n1.2
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

@ nomp = 2
@ mpirank = 36 / $nomp
echo Running with $mpirank MPI ranks with $nomp OpenMP threads per rank 
time srun -n $mpirank -c $nomp lmp -sf omp -pk omp $nomp -in large.in >& md.log


