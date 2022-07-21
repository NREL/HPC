#!/bin/bash
#SBATCH --job-name=vasp_gpu
#SBATCH --time=1:00:00
#SBATCH --error=std.err
#SBATCH --output=std.out
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --account=hpcapps
#SBATCH --nodes=1
#SBATCH --gpu-bind=map_gpu:0,1

#To run on multiple nodes, change the last two SBATCH lines:
#SBATCH --nodes=4
#SBATCH --gpu-bind=map_gpu:0,1,0,1,0,1,0,1 #one set of "0,1" per node

module purge

#Load the OpenACC GPU build and necessary modules
module use /nopt/nrel/apps/220511a/modules/lmod/linux-centos7-x86_64/gcc/12.1.0
ml fftw nvhpc
export LD_LIBRARY_PATH=/nopt/nrel/apps/220511a/install/opt/spack/linux-centos7-skylake_avx512/gcc-12.1.0/nvhpc-22.3-c4qk6fly5hls3mjimoxg6vyuy5cc3vti/Linux_x86_64/22.3/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/nopt/nrel/apps/220511a/install/opt/spack/linux-centos7-skylake_avx512/gcc-12.1.0/nvhpc-22.3-c4qk6fly5hls3mjimoxg6vyuy5cc3vti/Linux_x86_64/22.3/compilers/extras/qd/lib:$LD_LIBRARY_PATH
export PATH=/projects/hpcapps/tkaiser2/vasp/6.3.1/nvhpc_acc:$PATH

mpirun -npernode 2 vasp_std &> out
