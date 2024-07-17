# AMR-Wind

AMR-Wind is a massively parallel, block-structured adaptive-mesh,
incompressible flow solver for wind turbine and wind farm
simulations. The primary applications for AMR-Wind are: performing
large-eddy simulations (LES) of atmospheric boundary layer (ABL)
flows, simulating wind farm turbine-wake interactions using actuator
disk or actuator line models for turbines, and as a background solver
when coupled with a near-body solver (e.g., Nalu-Wind) with overset
methodology to perform blade-resolved simulations of multiple wind
turbines within a wind farm. For more information see [the AMR-Wind documentation](https://github.com/Exawind/amr-wind).

AMR-Wind is only supported on Kestrel. 


## Installation of AMR-Wind on GPU Nodes

AMR-wind can be installed by following the instructions [here](https://exawind.github.io/amr-wind/user/build.html#building-from-source).
On Kestrel GPU nodes, this can be achieved by first loading the following modules:

```bash
module restore 
source /nopt/nrel/apps/gpu_stack/env_cpe23.sh
ml gcc
ml PrgEnv-nvhpc
ml nvhpc/24.1
ml cray-libsci/22.12.1.1
ml cmake/3.27.9
ml python/3.9.13
```

Make sure the following modules are loaded using `module list`.

```bash
libfabric/1.15.2.0
craype-x86-genoa 
curl/8.6.0   
bzip2/1.0.8  
tar/1.34  
python/3.9.13
cray-dsmml/0.2.2 
cray-libsci/22.10.1.2 
gcc/10.1.0
craype-network-ofi  
nvhpc/24.1
cmake/3.27.9 
libxml2/2.10.3 
gettext/0.22.4 
craype/2.7.30 
cray-mpich/8.1.28 
PrgEnv-nvhpc/8.5.0
```


You can clone the latest version of AMR-wind from [here](https://github.com/Exawind/amr-wind?tab=readme-ov-file).
Once cloned, `cd` into the AMR directory and create a build folder.

You can create a file with the cmake instructions, 

```
vim conf_instructions
```

and copy the content below.

```
cmake .. -DAMR_WIND_ENABLE_CUDA=ON \
    -DAMReX_CUDA_ERROR_CAPTURE_THIS:BOOL=ON \
    -DCMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION:BOOL=ON \
    -DMPI_CXX_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpicxx \
    -DMPI_C_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpicc \
    -DMPI_Fortran_COMPILER=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3/bin/mpifort \
    -DAMReX_DIFFERENT_COMPILER=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DAMR_WIND_ENABLE_CUDA=ON \
    -DAMR_WIND_ENABLE_CUDA:BOOL=ON \
    -DAMR_WIND_ENABLE_OPENFAST:BOOL=OFF \
    -DAMR_WIND_ENABLE_NETCDF:BOOL=OFF \
    -DAMR_WIND_ENABLE_MPI:BOOL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMR_WIND_ENABLE_HYPRE:BOOL=OFF \
    -DAMR_WIND_ENABLE_MASA:BOOL=OFF \
    -DAMR_WIND_ENABLE_TESTS:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX:PATH=./install
```

You can execute the file using:

```
bash conf_instructions
```

Once the cmake step is done, you can:

```
make -j 
```

then 

```
make install -j 
```

You should now have a successful installation of AMR-Wind. 

At runtime, make sure to follow this sequence of module loads.

```
module restore 
source /nopt/nrel/apps/gpu_stack/env_cpe23.sh
ml PrgEnv-nvhpc
ml cray-libsci/22.12.1.1
```


## Running on the GPUs Using Modules

NREL makes available different modules for using AMR-Wind for CPUs and GPUs for
different toolchains. It is recommended that AMR-Wind be run on GPU nodes for obtaining the most optimal
performance.

Here is a sample script for submitting an AMR-Wind application run on multiple GPU nodes, with the user's input file and mesh grid in the current working directory.

??? example "Sample job script: Kestrel - Full GPU node"

    ```

    #!/bin/bash
    #SBATCH --time=1:00:00 
    #SBATCH --account=<user-account>
    #SBATCH --nodes=2
    #SBATCH --gpus=h100:4
    #SBATCH --exclusive
    #SBATCH --mem=0

    module restore 
    source /nopt/nrel/apps/gpu_stack/env_cpe23.sh
    module load PrgEnv-nvhpc
    module load amr-wind/main-craympich-nvhpc

    srun -K1 -n 16 --gpus-per-node=4 amr_wind abl_godunov-512.i >& ablGodunov-512.log

    ```
