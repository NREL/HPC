# OpenFAST and FAST.Farm

OpenFAST is a multi-physics, multi-fidelity tool for simulating the coupled dynamic response of wind turbines. Practically speaking, OpenFAST is the framework (or “glue code”) that couples computational modules for aerodynamics, hydrodynamics for offshore structures, control and electrical system (servo) dynamics, and structural dynamics to enable coupled nonlinear aero-hydro-servo-elastic simulation in the time domain. OpenFAST enables the analysis of a range of wind turbine configurations, including two- or three-blade horizontal-axis rotor, pitch or stall regulation, rigid or teetering hub, upwind or downwind rotor, and lattice or tubular tower. The wind turbine can be modeled on land or offshore on fixed-bottom or floating substructures.

FAST.Farm is a midfidelity multiphysics engineering tool for predicting the power performance and structural loads of wind turbines within a wind farm. FAST.Farm uses OpenFAST to solve the aero-hydro-servo-elastic dynamics of each individual turbine, but considers additional physics for wind farm-wide ambient wind in the atmospheric boundary layer; a wind-farm super controller; and wake deficits, advection, deflection, meandering, and merging. FAST.Farm is based on some of the principles of the dynamic wake meandering (DWM) model—including passive tracer modeling of wake meandering—but addresses many of the limitations of previous DWM implementations. FAST.Farm maintains low computational cost to support the often highly iterative and probabilistic design process.

 
## Building OpenFAST and FAST.Farm

In this section we provide cmake scripts for installation of OpenFAST and FAST.Farm. Both tools can be installed by following the instructions [here](https://openfast.readthedocs.io/en/main/source/install/index.html).

FAST.Farm is installed as a `cmake` option to the build of OpenFAST.

You can clone your desired verstion of OpenFAST from [here](https://github.com/OpenFAST/openfast). Once cloned, `cd` into the OpenFAST directory and create a `build` directory. Use the scripts given below from within the `build` directory to build OpenFAST and FAST.Farm. On a Kestrel CPU node, build OpenFAST by executing the following script from within the `build` directory:

??? example "Sample job script: Building OpenFAST and FAST.Farm using `cmake` on CPUs"
    ```
    #!/bin/bash

    module purge
    module load PrgEnv-intel/8.5.0
    module load intel-oneapi-mkl/2024.0.0-intel
    module load intel-oneapi
    module load binutils
    module load hdf5/1.14.3-intel-oneapi-mpi-intel

    module list

    cmake .. \
        -DCMAKE_Fortran_COMPILER=ifx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_FLAGS=-fPIC \
        -DCMAKE_C_FLAGS=-fPIC \
        -DBUILD_OPENFAST_CPP_API=ON \
        -DBUILD_FASTFARM=ON \
        -DDOUBLE_PRECISION:BOOL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENMP=ON \
        -DCMAKE_INSTALL_PREFIX:PATH=${PWD}/install

    nice make -j48
    make install
    ```

!!! Note
    The option `OPENFAST_CPP_API` needs to be enabled if further coupling to AMR-Wind is desired. Otherwise, the user can set the option to `OFF`.

!!! Note
    OpenFAST is not GPU-ready.


## Running OpenFAST

OpenFAST is a serial tool and can be executed by simply calling it directly
??? example "Sample job script: Running OpenFAST on a dedicated node"
```
    #!/bin/bash

    #SBATCH --account=<user-account> # Replace with your HPC account
    #SBATCH --time=01:00:00
    #SBATCH –-nodes=1
    #SBATCH --partition=shared

    module purge
    module load PrgEnv-intel/8.5.0
    module load intel-oneapi-mkl/2024.0.0-intel
    module load intel-oneapi
    module load binutils
    module load hdf5/1.14.3-intel-oneapi-mpi-intel

    openfast <your_turbine_input_file>.fst
```

Examples of turbine models are available in the regression tests repository, [here](https://github.com/OpenFAST/r-test/).

!!! Note
    OpenFAST input files change from one version to another. If you build a certain version, ensure your input files (or exemples from the regression tests repository linked above) are compatible with your version.


## Running FAST.Farm

FAST.Farm is OpenMP-capable, but still runs within a node. Its execution is similar to OpenFAST.

??? example "Sample job script: Running FAST.Farm on a dedicated node"
```
    #!/bin/bash

    #SBATCH --account=<user-account> # Replace with your HPC account
    #SBATCH --time=01:00:00
    #SBATCH –-nodes=1

    module purge
    module load PrgEnv-intel/8.5.0
    module load intel-oneapi-mkl/2024.0.0-intel
    module load intel-oneapi
    module load binutils
    module load hdf5/1.14.3-intel-oneapi-mpi-intel

    FAST.Farm <your_fastfarm_input_file>.fstf
```

FAST.Farm input deck can become complex depending on your simulation model. FAST.Farm users are encouraged to use accompaining toolbox for case setup available [here](https://github.com/OpenFAST/openfast_toolbox/tree/main/openfast_toolbox/fastfarm).
