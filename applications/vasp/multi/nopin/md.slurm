#!/bin/tcsh
#SBATCH --job-name m.nopin
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

#Run two job on one node
#Create two subdirectories, copy files, and delete slurm files 
cp -r ../../input run1 
rm run1/md.slurm 
echo NPAR = 2 >> run1/INCAR
echo NSIM = 16 >> run1/INCAR
cp -r run1 run2

setenv OMP_NUM_THREADS 1

#Run two jobs simulataneously
cd run1
time srun -n 18 vasp_std >& md.log &
cd ../run2
time srun -n 18 vasp_std >& md.log &

#Wait to complete
wait
echo All Done.

