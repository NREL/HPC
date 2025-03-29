# ExaWind
[`ExaWind`](https://www.nrel.gov/docs/fy21osti/80401.pdf) is a suite of applications that simulate wind turbines and wind farms on accelerated systems. The applications include AMR-Wind, Nalu-Wind, TIOGA, and OpenFAST. [`AMR-Wind`](https://github.com/Exawind/amr-wind) is a massively parallel, block-structured adaptive-mesh, incompressible flow solver for wind turbine and wind farm simulations. [`Nalu-Wind`](https://exawind.github.io/nalu-wind/index.html) is a generalized, unstructured, massively parallel, incompressible flow solver for wind turbine and wind farm simulations. [`TIOGA`](https://github.com/jsitaraman/tioga) is a library for overset grid assembly on parallel distributed systems. [`OpenFAST`](https://openfast.readthedocs.io/en/main/) is a multi-physics, multi-fidelity tool for simulating the coupled dynamic response of wind turbines.

## Building ExaWind 
We recommend installing ExaWind packages, either coupled or standalone, via [`exawind-manager`](https://github.com/Exawind/exawind-manager). While ExaWind-manager is recommended, CMake could be used as a substitute for installing the necessary packages. Instructions for building ExaWind packages with ExaWind-Manager and AMR-Wind/OpenFAST with CMake are described below.

### Building ExaWind using ExaWind-manager on Kestrel-CPU
The following examples demonstrate how to use ExaWind-manager for building common ExaWind applications on Kestrel. The build requires a compute node having at least 36 cores and using Intel or GNU compiler. To avoid space and speed issues, clone ExaWind-manager to scratch, not your home directory; then activate it, create and activate a Spack environment, and finally concretize and build. When making a Spack environment, you can add (+) or remove (-) specs, and adjust versions (@) for the main and dependent (^) applications. The first example outlines the process of building ExaWind using the master branch, omitting GPU functionalities and the AMR-Wind and Nalu-Wind as its dependencies. The second example outlines the process of building a coupled release of AMR-Wind and OpenFAST from the develop branch. The final two examples illustrate how to build the released AMR-Wind and Nalu-Wind versions.

??? example "Building `ExaWind`"
    ```
    $ salloc --time=01:00:00 --account= <project account> --partition=shared --nodes=1 --ntasks-per-node=36

    # Intel
    $ module load PrgEnv-intel
    $ module load cray-mpich/8.1.28
    $ module load cray-libsci/23.12.5
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager
    
    # Activate exawind-manager
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name exawind-cpu --spec 'exawind@master~amr_wind_gpu~cuda~gpu-aware-mpi~nalu_wind_gpu ^amr-wind@main~cuda~gpu-aware-mpi+hypre+mpi+netcdf+openmp+shared ^nalu-wind@master~cuda~fftw~gpu-aware-mpi+hypre+shared ^tioga@develop %oneapi'

    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-cpu

    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install

    ```

??? example "Building Coupled `AMR-Wind` and `OpenFAST`"
    ```
    $ salloc --time=01:00:00 --account= <project account> --partition=shared --nodes=1 --ntasks-per-node=52

    # Intel
    $ module load PrgEnv-intel
    $ module load cray-mpich/8.1.28
    $ module load cray-libsci/23.12.5
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager
    
    # Activate exawind-manager
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name amrwind-openfast-cpu --spec 'amr-wind+hypre+netcdf+openfast+openmp ^openfast@develop+openmp+rosco %oneapi'

    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/amrwind-openfast-cpu
     
    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install
    
    ```

??? example "Building `AMR-Wind`"
    ```
    $ salloc --time=01:00:00 --account= <project account> --partition=shared --nodes=1 --ntasks-per-node=52

    # Intel
    $ module load PrgEnv-intel
    $ module load cray-mpich/8.1.28
    $ module load cray-libsci/23.12.5
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager
    
    # Activate exawind-manager	
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name amrwind-cpu --spec 'amr-wind+hypre+netcdf+openmp %oneapi'

    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/amrwind-cpu

    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install

    ```

??? example "Building `Nalu-Wind`"
    ```
    $ salloc --time=01:00:00 --account= <project account> --partition=shared --nodes=1 --ntasks-per-node=52

    # Intel
    $ module load PrgEnv-intel
    $ module load cray-mpich/8.1.28
    $ module load cray-libsci/23.12.5
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager
    
    # Activate exawind-manager
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name naluwind-cpu --spec 'nalu-wind+hypre+netcdf+openmp %oneapi'
      
    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/naluwind-cpu
    
    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install

    ```

### Building AMR-Wind and OpenFAST using CMake on Kestrel-CPU
This section describes how to install AMR-Wind and the coupled AMR-Wind/OpenFAST using provided CMake scripts. AMR-wind can be installed by following the instructions [here](https://exawind.github.io/amr-wind/user/build.html#building-from-source). You can clone your desired version of AMR-wind from [here](https://github.com/Exawind/amr-wind). Once cloned, `cd` into the AMR-Wind directory and create a `build` directory. Use the scripts given below from within the `build` directory to build AMR-Wind. On a Kestrel CPU node, build AMR-Wind for CPUs by executing the following script from within the `build` directory:

??? example "Sample job script: Building AMR-Wind using `cmake` on Kestrel-CPU"
    ```
    #!/bin/bash

    module purge
    module load PrgEnv-intel
    module load netcdf/4.9.2-intel-oneapi-mpi-intel
    module load netlib-scalapack/2.2.0-gcc
    export LD_LIBRARY_PATH=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel:$LD_LIBRARY_PATH
    export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12
    export MPICH_VERSION_DISPLAY=1
    export MPICH_ENV_DISPLAY=1
    export MPICH_OFI_CXI_COUNTER_REPORT=2
    export FI_MR_CACHE_MONITOR=memhooks
    export FI_CXI_RX_MATCH_MODE=software
    export MPICH_SMP_SINGLE_COPY_MODE=NONE

    echo $LD_LIBRARY_PATH |tr ':' '\n'

    module list

    cmake .. \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DMPI_Fortran_COMPILER=mpifort \
        -DCMAKE_Fortran_COMPILER=ifx \
        -DAMR_WIND_ENABLE_CUDA:BOOL=OFF \
        -DAMR_WIND_ENABLE_MPI:BOOL=ON \
        -DAMR_WIND_ENABLE_OPENMP:BOOL=OFF \
        -DAMR_WIND_TEST_WITH_FCOMPARE:BOOL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMR_WIND_ENABLE_NETCDF:BOOL=ON \
        -DAMR_WIND_ENABLE_HYPRE:BOOL=OFF \
        -DAMR_WIND_ENABLE_MASA:BOOL=OFF \
        -DAMR_WIND_ENABLE_TESTS:BOOL=ON \
        -DAMR_WIND_ENABLE_ALL_WARNINGS:BOOL=ON \
        -DBUILD_SHARED_LIBS:BOOL=ON \
        -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

    nice make -j32
    make install

    ```

If coupling to OpenFAST is needed, an additional flag must be passed to `cmake`. A complete example is given below.

??? example "Sample job script: Building AMR-Wind coupled to OpenFAST using `cmake` on Kestrel-CPU"
    ```
    #!/bin/bash

    openfastpath=/full/path/to/your/openfast/build/install

    module purge
    module load PrgEnv-intel
    module load netcdf/4.9.2-intel-oneapi-mpi-intel
    module load netlib-scalapack/2.2.0-gcc
    export LD_LIBRARY_PATH=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel:$LD_LIBRARY_PATH
    export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12
    export MPICH_VERSION_DISPLAY=1
    export MPICH_ENV_DISPLAY=1
    export MPICH_OFI_CXI_COUNTER_REPORT=2
    export FI_MR_CACHE_MONITOR=memhooks
    export FI_CXI_RX_MATCH_MODE=software
    export MPICH_SMP_SINGLE_COPY_MODE=NONE

    echo $LD_LIBRARY_PATH |tr ':' '\n'

    module list

    cmake .. \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DMPI_Fortran_COMPILER=mpifort \
        -DCMAKE_Fortran_COMPILER=ifx \
        -DAMR_WIND_ENABLE_CUDA:BOOL=OFF \
        -DAMR_WIND_ENABLE_MPI:BOOL=ON \
        -DAMR_WIND_ENABLE_OPENMP:BOOL=OFF \
        -DAMR_WIND_TEST_WITH_FCOMPARE:BOOL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMR_WIND_ENABLE_NETCDF:BOOL=ON \
        -DAMR_WIND_ENABLE_HYPRE:BOOL=OFF \
        -DAMR_WIND_ENABLE_MASA:BOOL=OFF \
        -DAMR_WIND_ENABLE_TESTS:BOOL=ON \
        -DAMR_WIND_ENABLE_ALL_WARNINGS:BOOL=ON \
        -DBUILD_SHARED_LIBS:BOOL=ON \
        -DOpenFAST_ROOT:PATH=${openfastpath} \
        -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

    nice make -j32
    make install

    ```

### Building ExaWind with ExaWind-manager on Kestrel-GPU
Building ExaWind applications on GPUs uses processes similar to those on CPUs described earlier. Example one details building the released ExaWind GPU version with its dependencies (AMR-Wind and Nalu-Wind), while examples two and three build the released AMR-Wind and Nalu-Wind independently.

??? example "Building `ExaWind` on GPUs"
    ```
    $ salloc --time=02:00:00 --account=hpcapps --partition=gpu-h100 --gres=gpu:h100:1 --nodes=1 --ntasks-per-node=52 --gres=gpu:h100:1

    # GNU
    $ module load PrgEnv-gnu
    $ module load cray-mpich/8.1.28
    $ module load  cray-libsci/23.12.5
    $ module load cuda
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager

    # Activate exawind-manager
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name exawind-gpu --spec 'exawind+cuda+gpu-aware-mpi+amr_wind_gpu+nalu_wind_gpu cuda_arch=90 %gcc'

    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-gpu

    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install
    
    ```

??? example "Building `AMR-Wind` on GPUs"
    ```
    $ salloc --time=02:00:00 --account=hpcapps --partition=gpu-h100 --gres=gpu:h100:1 --nodes=1 --ntasks-per-node=52 --gres=gpu:h100:1

    # GNU
    $ module load PrgEnv-gnu
    $ module load cray-mpich/8.1.28
    $ module load  cray-libsci/23.12.5
    $ module load cuda
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager

    # Activate exawind-manager
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name amrwind-gpu --spec 'amr-wind+cuda+gpu-aware-mpi+hypre+netcdf+hdf5 cuda_arch=90  %gcc'

    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/amrwind-gpu

    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install
    
    ```

??? example "Building `Nalu-Wind` on GPUs"
    ```
    $ salloc --time=02:00:00 --account=hpcapps --partition=gpu-h100 --gres=gpu:h100:1 --nodes=1 --ntasks-per-node=52 --gres=gpu:h100:1

    # GNU
    $ module load PrgEnv-gnu
    $ module load cray-mpich/8.1.28
    $ module load  cray-libsci/23.12.5
    $ module load cuda
    $ module load cray-python

    # clone ExaWind-manager
    $ cd /scratch/${USER}
    $ git clone --recursive https://github.com/Exawind/exawind-manager.git
    $ cd exawind-manager

    # Activate exawind-manager
    $ export EXAWIND_MANAGER=`pwd`
    $ source ${EXAWIND_MANAGER}/start.sh && spack-start

    # Create Spack environment and change the software versions if needed
    $ mkdir environments
    $ cd environments
    $ spack manager create-env --name naluwind-gpu --spec 'nalu-wind+cuda+gpu-aware-mpi+hypre cuda_arch=90  %gcc'

    # Activate the environment
    $ spack env activate -d ${EXAWIND_MANAGER}/environments/naluwind-gpu

    # concretize specs and dependencies
    $ spack concretize -f

    # Build software
    $ spack -d install

    ```

### Building AMR-Wind Using CMake on Kestrel-GPU
On a Kestrel GPU node, build AMR-Wind for GPUs by executing the follow script from the `build` directory:

??? example "Sample job script: Building AMR-Wind using `cmake` on GPUs"
    ```
    #!/bin/bash

    module purge
    module load binutils
    module load PrgEnv-nvhpc
    module load cray-libsci/22.12.1.1
    module load cmake
    module load cmake/3.27.9
    module load cray-python
    module load netcdf-fortran/4.6.1-oneapi
    module load craype-x86-genoa
    module load craype-accel-nvidia90 
     
    export MPICH_GPU_SUPPORT_ENABLED=1
    export CUDAFLAGS="-L/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/lib -I/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/include -lhdf5 -lhdf5_hl -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_nvidia90} ${PE_MPICH_GTL_LIBS_nvidia90}"
    export CXXFLAGS="-L/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/lib -I/nopt/nrel/apps/gpu_stack/libraries-gcc/06-24/linux-rhel8-zen4/gcc-12.3.0/hdf5-1.14.3-zoremvtiklvvkbtr43olrq3x546pflxe/include -lhdf5 -lhdf5_hl -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_nvidia90} ${PE_MPICH_GTL_LIBS_nvidia90}"
     
    module list
     
    cmake .. \
        -DAMR_WIND_ENABLE_CUDA=ON \
        -DAMR_WIND_ENABLE_TINY_PROFILE:BOOL=ON \
        -DAMReX_CUDA_ERROR_CAPTURE_THIS:BOOL=ON \
        -DCMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION:BOOL=ON \
        -DCMAKE_CXX_COMPILER:STRING=CC \
        -DCMAKE_C_COMPILER:STRING=cc \
        -DMPI_CXX_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpicxx \
        -DMPI_C_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpicc \
        -DMPI_Fortran_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpifort \
        -DAMReX_DIFFERENT_COMPILER=ON \
        -DCMAKE_CUDA_ARCHITECTURES=90 \
        -DAMR_WIND_ENABLE_CUDA=ON \
        -DAMR_WIND_ENABLE_CUDA:BOOL=ON \
        -DAMR_WIND_ENABLE_OPENFAST:BOOL=OFF \
        -DAMR_WIND_ENABLE_NETCDF:BOOL=ON \
        -DAMR_WIND_ENABLE_HDF5:BOOL=ON \
        -DAMR_WIND_ENABLE_MPI:BOOL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMR_WIND_ENABLE_HYPRE:BOOL=OFF \
        -DAMR_WIND_ENABLE_MASA:BOOL=OFF \
        -DAMR_WIND_ENABLE_TESTS:BOOL=ON \
        -DCMAKE_INSTALL_PREFIX:PATH=./install
     
    make -j32 amr_wind

    ```

You should now have a successful installation of AMR-Wind. At runtime, make sure to load the same modules used to build, as discussed below.

## Running ExaWind

### Running ExaWind on Kestrel-CPU

ExaWind applications on more than 8 nodes run better on the [`hbw` ("high bandwidth") partition](../Systems/Kestrel/Running/index.md#high-bandwidth-partition) than on `short`, `standard`, or `long` partitions. **We strongly recommend submitting multi-node jobs to the `hbw` partition for the best performance and to save AUs**. Our benchmark studies show ExaWind applications perform best on the `hbw` partition with 72 MPI ranks per node. Following are example scripts illustrating the above recommendations.

!!! Note
    Single-node jobs are not allowed to be submitted to `hbw`; they should instead be submitted to `short`, `standard`, or `long`.

??? example "Sample job script: Running ExaWind with the `hbw` partition"
    ```
    #!/bin/bash
    
    #SBATCH --job-name=<job-name>
    #SBATCH --nodes=16
    #SBATCH --ntasks-per-node=72
    #SBATCH --time=1:00:00
    #SBATCH --partition=hbw
    #SBATCH --account=<account-name>
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-intel
    module load cray-python 
    module list

    export EXAWIND_MANAGER=/scratch/{user}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-cpu
    spack load exawind

    export MPICH_OFI_NIC_POLICY=NUMA

    # Adjust the ratio of total MPI ranks for AMR-Wind and Nalu-Wind as needed by a job 
    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) \
    --distribution=block:block --cpu_bind=rank_ldom exawind --awind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.25) \
    --nwind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.75) <input-name>.yaml

    ```

??? example "Sample job script: Running AMR-Wind using the `hbw` partition"
    ```
    #!/bin/bash

    #SBATCH --job-name=<job-name>
    #SBATCH --nodes=16
    #SBATCH --ntasks-per-node=72
    #SBATCH --time=1:00:00
    #SBATCH --partition=hbw
    #SBATCH --account=<account-name>
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-intel
    module load cray-mpich/8.1.28
    module load cray-libsci/23.12.5
    module load cray-python
    module list

    export EXAWIND_MANAGER=/scratch/{user}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/amrwind-cpu
    spack load amr-wind

    export MPICH_OFI_NIC_POLICY=NUMA

    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu_bind=rank_ldom amr_wind <input-name>.inp

    ```

??? example "Sample job script: Running Nalu-Wind using the `hbw` partition"
    ```
    #!/bin/bash

    #SBATCH --job-name=<job-name>
    #SBATCH --nodes=16
    #SBATCH --ntasks-per-node=72
    #SBATCH --time=1:00:00
    #SBATCH --partition=hbw
    #SBATCH --account=<account-name>
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-intel
    module load cray-mpich/8.1.28
    module load cray-libsci/23.12.5
    module load cray-python
    module list

    export EXAWIND_MANAGER=/scratch/{user}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/naluwind-cpu
    spack load nalu-wind

    export MPICH_OFI_NIC_POLICY=NUMA

    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu_bind=rank_ldom naluX <input-name>.yaml

    ```

Our benchmark studies suggest that using the [`stall` library](../Systems/Kestrel/Running/index.md#performancerecs) with the `hbw` partition could further improve ExaWind application performance. Moreover, **the stall library is highly recommended for ExaWind applications running on 8 nodes or less, each with 96 cores, across short, standard, or long partitions**. Optimizing the `MPICH_OFI_CQ_STALL_USECS` parameter is key to acheive the best performance. Following are sample scripts demonstrating the aforementioned recommendations.

??? example "Sample job script: Running ExaWind using the stall library"
    ```
    #!/bin/bash

    #SBATCH --job-name=<job-name>
    #SBATCH --partition=<partition-name> # hbw, short, standard	or long
    #SBATCH --nodes=<nodes> # >=16 nodes for hbw or <=8 nodes for short, standard or long 
    #SBATCH --ntasks-per-node=<cores> # 72 cores for hbw or 96 cores for short, standard or long
    #SBATCH --time=1:00:00
    #SBATCH --account=<account-name>
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-intel
    module load cray-mpich/8.1.28
    module load cray-libsci/23.12.5
    module load cray-python
    module list

    export EXAWIND_MANAGER=/scratch/{user}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-cpu
    spack load exawind

    export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12
    export MPICH_OFI_CQ_STALL=1
    # Find an optimal value from this list [1,3,6,9,12,16,20,24]
    export MPICH_OFI_CQ_STALL_USECS=12
    export MPICH_OFI_CQ_MIN_PPN_PER_NIC=26
    export MPICH_OFI_NIC_POLICY=NUMA

    # Adjust the ratio of total MPI ranks for AMR-Wind and Nalu-Wind as needed by a job
    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) \
    --distribution=block:block --cpu_bind=rank_ldom exawind --awind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.25) \
    --nwind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.75) <input-name>.yaml

    ```
    
??? example "Sample job script: Running AMR-Wind using the stall library nodes"
    ```
    #!/bin/bash

    #SBATCH --job-name=<job-name>
    #SBATCH --partition=<partition-name> # hbw, short, standard or long
    #SBATCH --nodes=<nodes> # >=16 nodes for hbw or <=8 nodes for short, standard or long
    #SBATCH --ntasks-per-node=<cores> # 72 cores for hbw or 96 cores for short, standard or long
    #SBATCH --time=1:00:00
    #SBATCH --account=<account-name>
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-intel
    module load cray-mpich/8.1.28
    module load cray-libsci/23.12.5
    module load cray-python
    module list

    export EXAWIND_MANAGER=/scratch/{user}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/amrwind-cpu
    spack load amr-wind

    export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12
    export MPICH_OFI_CQ_STALL=1
    # Find an optimal value from this list [1,3,6,9,12,16,20,24]
    export MPICH_OFI_CQ_STALL_USECS=12
    export MPICH_OFI_CQ_MIN_PPN_PER_NIC=26
    export MPICH_OFI_NIC_POLICY=NUMA

    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu_bind=rank_ldom amr_wind <input-name>.inp

    ```
??? example "Sample job script: Running Nalu-Wind using the stall library nodes"
    ```
    #!/bin/bash

    #SBATCH --job-name=<job-name>
    #SBATCH --partition=<partition-name> # hbw, short, standard or long
    #SBATCH --nodes=<nodes> # >=16 nodes for hbw or <=8 nodes for short, standard or long
    #SBATCH --ntasks-per-node=<cores> # 72 cores for hbw or 96 cores for short, standard or long
    #SBATCH --time=1:00:00
    #SBATCH --account=<account-name>
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-intel
    module load cray-mpich/8.1.28
    module load cray-libsci/23.12.5
    module load cray-python
    module list

    export EXAWIND_MANAGER=/scratch/{user}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/naluwind-cpu
    spack load nalu-wind

    export LD_PRELOAD=/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpi_intel.so.12:/nopt/nrel/apps/cray-mpich-stall/libs_mpich_nrel_intel/libmpifort_intel.so.12
    export MPICH_OFI_CQ_STALL=1
    # Find an optimal value from this list [1,3,6,9,12,16,20,24]
    export MPICH_OFI_CQ_STALL_USECS=12
    export MPICH_OFI_CQ_MIN_PPN_PER_NIC=26
    export MPICH_OFI_NIC_POLICY=NUMA

    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu_bind=rank_ldom naluX <input-name>.yaml

    ```

### Running ExaWind on Kestrel-GPU

Running ExaWind on GPUs yields optimal performance. The following scripts illustrate how to submit jobs on the `gpu-h100` partition.

??? example "Sample job script: Running ExaWind on GPU nodes"
    ```
    #!/bin/bash

    #SBATCH --time=1:00:00 
    #SBATCH --account=<user-account>
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus=h100:4
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-gnu
    module load cray-mpich/8.1.28
    module load  cray-libsci/23.12.5
    module load cuda
    module load cray-python

    export EXAWIND_MANAGER=/scratch/${USER}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/exawind-gpu
    spack load exawind@master

    export MPICH_OFI_NIC_POLICY=NUMA

    # Adjust the ratio of total MPI ranks for AMR-Wind and Nalu-Wind as needed by a job 
    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu_bind=rank_ldom \
    exawind --nwind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.75) --awind $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES) * 0.25) <input-name>.yaml 
    wait

    ```

??? example "Sample job script: Running AMR-Wind on GPU nodes"
    ```
    #!/bin/bash

    #SBATCH --time=1:00:00
    #SBATCH --account=<user-account> # Replace with your HPC account
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus=h100:4
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-gnu
    module load cray-mpich/8.1.28
    module load  cray-libsci/23.12.5
    module load cuda
    module load cray-python

    export EXAWIND_MANAGER=/scratch/${USER}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/amrwind-gpu
    spack load amr-wind

    export MPICH_OFI_NIC_POLICY=NUMA

    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu-bind=rankldom amr_wind <input-name>.inp

    ```

??? example "Sample job script: Running Nalu-Wind on GPU nodes"
    ```
    #!/bin/bash

    #SBATCH --time=1:00:00
    #SBATCH --account=<user-account> 
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus=h100:4
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-gnu
    module load cray-mpich/8.1.28
    module load  cray-libsci/23.12.5
    module load cuda
    module load cray-python

    export EXAWIND_MANAGER=/scratch/${USER}/exawind-manager
    source ${EXAWIND_MANAGER}/start.sh && spack-start
    spack env activate -d ${EXAWIND_MANAGER}/environments/naluwind-gpu
    spack load nalu-wind

    export MPICH_OFI_NIC_POLICY=NUMA

    srun -N $SLURM_JOB_NUM_NODES -n $(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) --distribution=block:block --cpu-bind=rankldom naluX <input-name>.yaml

    ```
