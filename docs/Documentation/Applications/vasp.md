## About VASP

VASP computes an approximate solution to the many-body Schrödinger equation, either within density functional theory (DFT), solving the Kohn-Sham equations, or within the Hartree-Fock (HF) approximation, solving the Roothaan equations. Hybrid functionals that mix the Hartree-Fock approach with density functional theory are implemented as well. Furthermore, Green's functions methods (GW quasiparticles, and ACFDT-RPA) and many-body perturbation theory (2nd-order Møller-Plesset) are available in VASP.

In VASP, central quantities, like the one-electron orbitals, the electronic charge density, and the local potential are expressed in plane wave basis sets. The interactions between the electrons and ions are described using norm-conserving or ultrasoft pseudopotentials, or the projector-augmented-wave method.

To determine the electronic ground state, VASP makes use of efficient iterative matrix diagonalization techniques, like the residual minimization method with direct inversion of the iterative subspace (RMM-DIIS) or blocked Davidson algorithms. These are coupled to highly efficient Broyden and Pulay density mixing schemes to speed up the self-consistency cycle.

For further details, documentation, forums, and FAQs, see the [VASP website](https://www.vasp.at/).

## Accessing VASP on NREL's HPC clusters

The VASP license requires users to be a member of a "workgroup" defined by the University of Vienna or Materials Design. If you are receiving "Permission denied" errors when trying to use VASP, you must be made part of the "vasp" Linux group first. To join, please contact us with the following information:

- Your name
- The workgroup PI
- Whether you are licensed through Vienna (academic) or Materials Design, Inc. (commercial)
- If licensed through Vienna:
  - the e-mail address under which you are registered with Vienna as a workgroup member (this may not be the e-mail address you used to get an HPC account)
  - Your VASP license ID
- If licensed through Materials Design,
  - proof of current licensed status

Once status can be confirmed, we can provide access to our VASP builds on Eagle, Swift or Vermilion. 

## Supported Versions

NREL offers support for VASP 5 and VASP 6 on CPUs, and a GPU build on Eagle and Vermilion as well.
- For CPUs, VASP can be built with either Intel compilers/Intel MPI or GNU compilers/Open MPI. In general the Intel MPI builds are faster and are reoccommended over the Open MPI builds.
- For GPUs, VASP can be built with Cuda or with OpenACC. The OpenACC GPU-port of VASP was released with VASP 6.2.0, and the Cuda GPU-port of VASP was dropped in VASP 6.3.0. The OpenACC build shows significant performance improvement compared to the Cuda build, but is more susceptible to running out of memory. 

|                    |     Eagle     |     Swift     |   Vermilion   |
| ------------------ | ------------- | ------------- | ------------- |
| VASP 5 (Intel MPI) |       x       |               |       x       |
| VASP 6 (Intel MPI) |       x       |       x       |       x       |
| VASP 6 (Open MPI)  |       x       |       x       |       x       |
| GPU VASP (Cuda)    |       x       |               |       x       |
| GPU VASP (OpenAcc) |       x       |               |       x       | 

Three distinct executables have been made available with each build of VASP:

1. vasp_gam is for Gamma-point-only runs typical for large unit cells;

2. vasp_std is for general k-point meshes with collinear spins; and,

3. vasp_ncl is for general k-point meshes with non-collinear spins.

```vasp_gam``` and ```vasp_std``` builds include the alternative optimization and [transition state theory tools from University of Texas-Austin](http://theory.cm.utexas.edu/vtsttools/) developed by Graeme Henkelman's group, and [implicit solvation models from the University of Florida](http://vaspsol.mse.ufl.edu/) developed by Mathew and Hennig.

## Getting Started

VASP is available through modules on Eagle, Swift and Vermilion. Use the command ```module avail vasp``` to view the versions of VASP available on each cluster. The module marked with "(D)" is the default module - slurm will chose to load this module if you run the command ```module load vasp```. To load other vasp modules, include the entire module name in the module load command, such as ```module load vasp/5.4.4_centos77```.

On Eagle, you should see:
```
[user@el1 ~]$ module avail vasp

------------------------------------------------------------------- /nopt/nrel/apps/modules/test/modulefiles --------------------------------------------------------------------
   vasp/5.4.4_raptor    vasp/6.1.0    vasp/6.1.1-openmpi    vasp/6.3.1-nvhpc_acc (D)

------------------------------------------------------------------ /nopt/nrel/apps/modules/default/modulefiles ------------------------------------------------------------------
   vasp/5.4.4_centos77    vasp/6.1.1    vasp/6.1.2

```

On Swift, you should see:
```
[clarson@swift-login-1 ~]$ module avail vasp

------------------------------------------ /nopt/nrel/apps/210928a/level02/modules/lmod/linux-rocky8-x86_64/openmpi/4.1.1-mkxx6h3/Core ------------------------------------------
   vasp/6.1.1-qb5uihl

----------------------------------------------------- /nopt/nrel/apps/210928a/level01/modules/lmod/linux-rocky8-x86_64/Core -----------------------------------------------------
   vaspintel/1.0-mkr32z6
```

On Vermilion, you should see:
```
[clarson@vs-login-1 ~]$ module avail vasp

------------------------------------------------------------------- /nopt/nrel/apps/modules/test/modulefiles --------------------------------------------------------------------
   vasp/5.4.4    vasp/6.1.1-openmpi    vasp/6.3.1-nvhpc_acc    vasp/6.3.1 (D)
```

### Input Files

To run VASP, 4 input files are needed: POSCAR, POTCAR, INCAR, KPOINTS. For more information about VASP input files, see the [VASP wiki](https://www.vasp.at/wiki/index.php/Input).

NREL's benchmark repositry contains input files for two VASP benchmarks. Benchmark 1 represents a typical high-accuracy band structure calculation, involving bootstrapping from an initial approximate GGA wavefunction, through a hybrid HSE function, to a final band structure from a GW calculation. Benchmark 2 represents a typical surface catalysis study calculation, with large unit cell and k-point sampling limited to the Gamma point. A single model chemistry (DFT functional and plane-wave basis) is employed, and strong scaling with respect to MPI rank count is of interest. The input files can be accessed in the [bench1](https://github.com/NREL/ESIFHPC3/tree/master/VASP/bench1/input) and [bench2](https://github.com/NREL/ESIFHPC3/tree/master/VASP/bench2/input) folders.

## Example Job Scripts And Performance Recommendations

### For Eagle

Performance recommendations for Eagle here: https://github.com/NREL/HPC/tree/master/applications/vasp/Performance%20Study%202
Also another performance study here: https://github.com/NREL/HPC/tree/master/applications/vasp/Performance%20Study%201

??? example "Eagle: VASP 6 (Intel MPI) on CPUs"

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --nodes=1

module purge

#Load Intel MPI VASP build
ml vasp

srun -n 36 vasp_std &> out
```

??? example "Eagle: VASP 6 (Open MPI) on CPUs"

```
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

srun -n 36 vasp_std &> out
```

??? example "Eagle: VASP 5 (Intel MPI) on CPUs"

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --nodes=1

module purge

#Load Intel MPI VASP 5 build
ml vasp/5.4.4_raptor

srun -n 36 vasp_std &> out
```

??? example "Eagle: VASP 6 (OpenACC) on GPUs"

```
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
```

??? example "Eagle: VASP 6 (Cuda) on GPUs"

To run the Cuda build of VASP on Eagle's GPUs, we can call the ```vasp_gpu``` exectuable in a module for a build of VASP older than 6.3.0. To use both GPUs per node, make sure to set ```#SBATCH --gres=gpu:2``` and ```#SBATCH --ntasks-per-node=2```.

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=short
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 
#SBATCH --gres=gpu:2

module purge

#Load Intel MPI VASP build
ml vasp

srun -n 36 vasp_gpu &> out
```

### For Swift

Performance recommendations for Swift here: https://github.com/NREL/HPC/tree/master/applications/vasp/Performance%20Study%202 

??? example "Swift: VASP 6 (Intel MPI) on CPUs"
```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=test
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1

#Set --exclusive if you would like to prevent any other jobs from running on the same nodes (including your own)
#You will be charged for the full node regardless of the fraction of CPUs/node used
#SBATCH --exclusive

module purge

#Load Intel MPI VASP build and necessary modules
ml vaspintel 
ml slurm/21-08-1-1-o2xw5ti 
ml gcc/9.4.0-v7mri5d 
ml intel-oneapi-compilers/2021.3.0-piz2usr 
ml intel-oneapi-mpi/2021.3.0-hcp2lkf 
ml intel-oneapi-mkl/2021.3.0-giz47h4

srun -n 64 vasp_std &> out
```

??? example "Swift: VASP 6 (Intel MPI) on CPUs: run multiple jobs on the same node(s)"

The following script launches two instances of ```srun vasp_std``` on the same node using an array job. Each job will be constricted to 32 cores on the node. 

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=parallel
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1

#Set --exclusive=user if you would like to prevent anyone else from running on the same nodes as you
#You will be charged for the full node regardless of the fraction of CPUs/node used
#SBATCH --exclusive=user

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

#Load Intel MPI VASP build and necessary modules
ml vaspintel 
ml slurm/21-08-1-1-o2xw5ti 
ml gcc/9.4.0-v7mri5d 
ml intel-oneapi-compilers/2021.3.0-piz2usr 
ml intel-oneapi-mpi/2021.3.0-hcp2lkf 
ml intel-oneapi-mkl/2021.3.0-giz47h4

srun -n 32 vasp_std &> out
```

??? example "Swift: VASP 6 (Intel MPI) on CPUs: run a single job on a node shared with other users"

The following script launches ```srun vasp_std``` on only 32 cores on a single node. The other 32 cores remain open for other users to use. You will only be charged for half of the node hours. 

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=test
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1

#To make sure that you are only being charged for the CPUs your job is using, set mem=2GB*CPUs/node
#--mem sets the memory used per node
#SBATCH --mem=64G

module purge

#Load Intel MPI VASP build and necessary modules
ml vaspintel 
ml slurm/21-08-1-1-o2xw5ti 
ml gcc/9.4.0-v7mri5d 
ml intel-oneapi-compilers/2021.3.0-piz2usr 
ml intel-oneapi-mpi/2021.3.0-hcp2lkf 
ml intel-oneapi-mkl/2021.3.0-giz47h4

srun -n 32 vasp_std &> out
```

??? example "Swift: VASP 6 (Open MPI) on CPUs"
```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=test
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1

#Set --exclusive if you would like to prevent any other jobs from running on the same nodes (including your own)
#You will be charged for the full node regardless of the fraction of CPUs/node used
#SBATCH --exclusive

module purge

#Load OpenMPI VASP build and necessary modules
ml vasp 
ml slurm/21-08-1-1-o2xw5ti 
ml openmpi/4.1.1-6vr2flz

srun -n 64 vasp_std &> out
```

??? example "Swift: VASP 6 (Open MPI) on CPUs: run multiple jobs on the same node(s)"

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=parallel
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1

#Set --exclusive=user if you would like to prevent anyone else from running on the same nodes as you
#You will be charged for the full node regardless of the fraction of CPUs/node used
#SBATCH --exclusive=user

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

srun -n 32 vasp_std &> out
```

??? example "Swift: VASP 6 (Open MPI) on CPUs: run a single job on a node shared with other users"

The following script launches ```srun vasp_std``` on only 32 cores on a single node. The other 32 cores remain open for other users to use. You will only be charged for half of the node hours. 

```
#!/bin/bash
#SBATCH --job-name="benchmark"
#SBATCH --account=hpcapps
#SBATCH --partition=test
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1

#To make sure that you are only being charged for the CPUs your job is using, set mem=2GB*CPUs/node
#--mem sets the memory used per node
#SBATCH --mem=64G

module purge

#Load OpenMPI VASP build and necessary modules
ml vasp 
ml slurm/21-08-1-1-o2xw5ti 
ml openmpi/4.1.1-6vr2flz

srun -n 32 vasp_std &> out
```

### Vermilion

VASP runs faster on 1 node than on 2 nodes. In some cases, VASP runtimes on 2 nodes have been observed to be double (or more) the run times on a single node. Many issues have been reported for running VASP on multiple nodes, especially when requesting all available cores in each node. In order for MPI to work reliably on Vermilion, it is necessary to specify the interconnect network that Vermilion should use to communicate between nodes. This is documented in each of the scripts below. Different solutions exists for Open MPI and Intel MPI. The documented recommendations for setting the interconnect network have been shown to work well for multi-node jobs on 2 nodes, but aren't guaranteed to produce succesful multi-node runs on 4 nodes. The Open MPI multi-node jobs are more reliable on Vermilion, but Intel MPI VASP jobs show better runtime performance. 

If your multi-node Intel MPI VASP job is crashing on Vermilion, try replacing your srun line with the following mpirun run line. ```-iface ens7``` sets ens7 as the interconnect. 
```
I_MPI_OFI_PROVIDER=tcp mpirun -iface ens7 -np 16 vasp_std
```

If your multi-node Open MPI VASP job is crashing on Vermilion, replace a call to load an openmpi module with the following lines. The OMPI_MCA_param variable sets ens7 as the interconnect. 
```
module use /nopt/nrel/apps/220525b/level01/modules/lmod/linux-rocky8-x86_64/gcc/12.1.0
module load openmpi
OMPI_MCA_param="btl_tcp_if_include ens7"
```

Due to low performance and reliability of multi-node VASP jobs on Vermilion, it is recommended to run VASP on a singe node. If many cores are needed for your VASP calcualtion, run your job in the lg partition (60 cores/node), which provides the largest numbers of cores per node.

??? example "Vermilion: VASP 6 (Intel MPI) on CPUs"

??? example "Vermilion: VASP 6 (Open MPI) on CPUs"

??? example "Vermilion: VASP 5 (Intel MPI) on CPUs"

??? example "Vermilion: VASP 6 (OpenACC) on GPUs"

??? example "Vermilion: VASP 6 (Cuda) on GPUs"

## Troubleshooting

VASP known issues
- VASP on Vermilion can crash on multi-node jobs (see workarounds in Vermilion section) - OpenMPI more stable for multi-node jobs
- OpenACC build eats up more memory but is faster (Cuda build doesn't eat up as much memory)
- VASP 6.1.1 (or maybe it's 6.1.2?) crashes hse calculations (due to specific version of compiler and vasp) - use a newer version instead

## Perfmarmance Recommendations

