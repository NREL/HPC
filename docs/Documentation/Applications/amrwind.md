# AMR-Wind

AMR-Wind is a massively parallel, block-structured adaptive-mesh, incompressible flow solver for wind turbine and wind farm simulations. The primary applications for AMR-Wind are: performing large-eddy simulations (LES) of atmospheric boundary layer (ABL) flows, simulating wind farm turbine-wake interactions using actuator disk or actuator line models for turbines, and as a background solver when coupled with a near-body solver (e.g., Nalu-Wind) with overset methodology to perform blade-resolved simulations of multiple wind turbines within a wind farm. For more information see [the AMR-Wind documentation](https://github.com/Exawind/amr-wind).

In this page, we provide instructions for building AMR-Wind using Exawind-Manager or CMake, both on the CPU and GPU. We also give examples on how to run AMR-Wind.

## Building AMR-Wind

### Using Exawind-Manager

The [`exawind-manager`](https://github.com/Exawind/exawind-manager) is a dev-ops tooling and configuration management for ExaWind development. It is the officially supported compilation process. 

The following script builds AMR-Wind (version 3.4.0 for illustration purposes) with support for NetCDF and HDF5:

??? example "Sample job script: Building AMR-Wind using `exawind-manager`"
    ```
    #!/bin/bash
    #SBATCH -o %x.o%j
    #SBATCH --account=<user-account> # Replace with your HPC account
    #SBATCH --time=60:00
    #SBATCH --nodes=1

    git clone --recursive https://github.com/Exawind/exawind-manager.git
    cd exawind-manager
    source shortcut.sh
    mkdir environments
    cd environments
    quick-create-dev -n amr-wind-env -s amr-wind+netcdf+hdf5+tiny_profile@3.4.0
    spack concretize -f
    spack env depfile -o Makefile
    make -j64
    ```

If coupling to OpenFAST is needed, add `openfast` to the `quick-create-dev` call above:
```
quick-create-dev -n amr-wind-env -s amr-wind+netcdf+hdf5+openfast+tiny_profile@3.4.0
```

Additional details and troubleshooing information are available on their documentation, available [here](https://exawind.github.io/amr-wind/walkthrough/compiling.html).



### Using `cmake`

In this section we provide cmake scripts for installation of AMR-Wind. AMR-wind can be installed by following the instructions [here](https://exawind.github.io/amr-wind/user/build.html#building-from-source).

You can clone your desired verstion of AMR-wind from [here](https://github.com/Exawind/amr-wind). Once cloned, `cd` into the AMR-Wind directory and create a `build` directory. Use the scripts given below from within the `build` directory to build AMR-Wind.

#### Installation of AMR-Wind on CPU Nodes
On a Kestrel CPU node, build AMR-Wind for CPUs by executing the following script from within the `build` directory:

??? example "Sample job script: Building AMR-Wind using `cmake` on CPUs"
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

??? example "Sample job script: Building AMR-Wind coupled to OpenFAST using `cmake` on CPUs"
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


### Installation of AMR-Wind on GPU Nodes

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

You should now have a successful installation of AMR-Wind. 

At runtime, make sure to load the same modules used to build, as discussed below.


## Running AMR-Wind

### Running AMR-Wind on the CPU nodes

On Kestrel, AMR-Wind performs the best on CPU nodes in the [`hbw` ("high bandwidth") partition](../Systems/Kestrel/Running/index.md#high-bandwidth-partition), which each have two network interface cards (NICs). **We strongly recommend submitting multi-node AMR-Wind jobs to the `hbw` partition for the best performance and to save AUs** when compared to running on single-NIC nodes in `short`, `standard`, or `long`. 

Additionally, according to benchmarks, AMR-Wind achieves the best performance on Kestrel CPU nodes using 72 MPI ranks per node. An example script using the current CPU module of AMR-Wind using all of the best practice recommendations is provided below.

!!! Note
    Single-node jobs are not allowed to be submitted to `hbw`; they should instead be continued to be submitted to the "general" CPU partitions such as `short`, `standard`, or `long`.

??? example "Sample job script: Running AMR-Wind on high-bandwidth `hbw` nodes"
    ```
    #!/bin/bash​

    #SBATCH --account=<user-account> # Replace with your HPC account
    #SBATCH –-partition=hbw​
    #SBATCH --time=01:00:00
    #SBATCH –-nodes=16 # May need to change depending on your problem​​

    export FI_MR_CACHE_MONITOR=memhooks​
    export FI_CXI_RX_MATCH_MODE=software​
    export MPICH_SMP_SINGLE_COPY_MODE=NONE​
    export MPICH_OFI_NIC_POLICY=NUMA​

    # Optimal number of launcher (srun) tasks per node benchmarked on Kestrel
    export SRUN_TASKS_PER_NODE=72

    # Replace <input>​ with your input file
    srun -N $SLURM_JOB_NUM_NODES \
        -n $(($SRUN_TASKS_PER_NODE * $SLURM_JOB_NUM_NODES)) \
        --ntasks-per-node=$SRUN_TASKS_PER_NODE \
        --distribution=block:block \
        --cpu_bind=rank_ldom \
        amr_wind <input>​
    ```

### Running AMR-Wind on the GPU nodes

A module for AMR-Wind can also be run on GPU nodes, which can obtain the most optimal performance.

Here is a sample script for submitting an AMR-Wind application run on multiple GPU nodes, with the user's input file and mesh grid in the current working directory.

??? example "Sample job script: Running AMR-Wind on GPU nodes"
    ```
    #!/bin/bash

    #SBATCH --time=1:00:00 
    #SBATCH --account=<user-account> # Replace with your HPC account
    #SBATCH --nodes=2
    #SBATCH --gpus=h100:4
    #SBATCH --exclusive
    #SBATCH --mem=0

    module load PrgEnv-nvhpc
    module load amr-wind/main-craympich-nvhpc

    # Replace <input>​ with your input file
    srun -K1 -n 16 --gpus-per-node=4 amr_wind <input>

    ```

