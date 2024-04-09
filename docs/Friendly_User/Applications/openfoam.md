OpenFoam installation 
=====================


Building openfoam with cray-mpich and gcc 
---------------------------------

Instruction for the installing OpenFoam are available [here](https://openfoam.org/download/source/).

In the instruction, you will be cloning the OpenFoam folder which we will refere to as `$OPENFOAM`.

In order to build Openfoam with cray-mpich, 2 files need to be edited.

1- `$OPENFOAM/etc/bashrc`

In this file, the variable `WM_MPLIB` will be defined as `MPICH`. 
Searhc for the line where the varibale is exported and replace it with 

```
export WM_MPLIB=MPICH
```

2- `$OPENFOAM/etc/config.sh/mpi`

This file defines where mpich is defined on the system. 
You will search for the Mpich definition block and replace it with 

```
    export MPI_ARCH_PATH=/opt/cray/pe/mpich/8.1.28/ofi/gnu/10.3
    export LD_LIBRARY_PATH="${MPI_ARCH_PATH}/lib:${LD_LIBRARY_PATH}"
    export PATH="${MPI_ARCH_PATH}/bin:${PATH}"
    export FOAM_MPI=mpich-8.1.28
    export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/gnu/10.3
    #export FOAM_MPI=mpich2-1.1.1p1
    #export MPI_HOME=$WM_THIRD_PARTY_DIR/$FOAM_MPI
    #export MPI_ARCH_PATH=$WM_THIRD_PARTY_DIR/platforms/$WM_ARCH$WM_COMPILER/$FOAM_MPI


    _foamAddPath    $MPI_ARCH_PATH/bin


    # 64-bit on OpenSuSE 12.1 uses lib64 others use lib
    _foamAddLib     $MPI_ARCH_PATH/lib$WM_COMPILER_LIB_ARCH
    _foamAddLib     $MPI_ARCH_PATH/lib


    _foamAddMan     $MPI_ARCH_PATH/share/man
    ;;
```

Before you install openfoam, make sure to load `Prgenv-gnu`.
This will load gcc and cray-mpich. 
Make sure the same module is loaded at runtime.


