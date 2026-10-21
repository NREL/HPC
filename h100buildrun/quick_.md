The slurm script below *quick* which does a build of C and Fortran MPI hello world.  The script shows that to get a MPI program to build with the standard Programing environments PrgEnv-{cray,intel,gnu}.  These environments build using Cray's MPI and the various back end compilers.  The script also builds for Intel MPI with Intel backend compilers and MPICH using gcc backends. By default the script will not run the C versions of the example.  You can enable running C by commenting out the line:
```
false ; TF=$?
```
You can get a copy of the script on Kestrel:
```
tar -xzf /nopt/nlr/apps/examples/cpu/quick.tgz
```

<!--
Or you can use git to do a download:

```
git clone $USER@kestrel.hpc.nlr.gov:/nopt/nlr/apps/examples/gpu/h100 h100
```
-->

This script will run on the CPU and GPU nodes.

The module sets used include:

* ml cpe-stack/25.03  PrgEnv-gnu
* ml cpe-stack/25.03  PrgEnv-intel
* ml cpe-stack/25.03  PrgEnv-nvidia
* ml oneapi/2025.3.1  intel-oneapi-mpi
* ml gcc/14.2.0 openmpi/5.0.3
* ml oneapi/2025.3.1 openmpi/5.0.3
* ml gcc mpich


??? example "quick"

    ```bash
    #!/bin/bash
    #SBATCH --time=0:10:00
    #SBATCH --partition=debug
    #SBATCH --nodes=1
    ##### Don't need gpus for this run
    ##SBATCH --gres=gpu:h100:4
    #SBATCH --exclusive
    #SBATCH --output=quick.out
    #SBATCH --error=quick.out
    
    # set the optimization flag for Cray MPI compiles
    # check to see that the submit node and partition match
    if  cat /proc/cpuinfo | grep AMD >/dev/null  ; then 
     ARCH=craype-x86-genoa
     if echo $SLURM_SUBMIT_HOST | egrep "kl5|kl6" >> /dev/null  ; 
     then 
     	echo running on AMD node ARCH= $ARCH
     else  
       echo You submitted from a CPU login node to a GPU node.
       echo Change the partition or run from login nodes "kl5 or kl6".
       exit 
     fi
    else
     ARCH=craype-x86-spr
     if echo $SLURM_SUBMIT_HOST | egrep "kl5|kl6" >> /dev/null  ; 
     then 
       echo You submitted from a GPU login node to a CPU node.
       echo Change the partition or run from login nodes "kl5 or kl6". 
    
       exit 
     else  
     	echo running on INTEL node ARCH= $ARCH
     fi
    fi
    
    # RUN THE C VERSION OF OUR PROGRAM
    true ; TF=$?
    false ; TF=$?
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; echo running c; else echo skipping c ;fi
    
    cmd () 
    { 
        echo "***********************" " $@";
        eval "$@"
    }
    
    cat > hellof.f90 << END
    !*********************************************
    !  This is a simple hello world program. Each processor 
    !  prints out its name, rank and  number of processors 
    !  in the current MPI run. 
    !*********************************************
          program hello
          use iso_fortran_env
          include "mpif.h"
          integer myid,numprocs,ierr,nlength
          character(len=MPI_MAX_LIBRARY_VERSION_STRING+1) :: version
          character (len=MPI_MAX_PROCESSOR_NAME+1):: myname
          call MPI_INIT( ierr )
          call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
          call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
          call MPI_Get_processor_name(myname,nlength,ierr)
          call MPI_Get_library_version(version, nlength, ierr)
          write (*,*) "Hello from ",trim(myname)," # ",myid," of ",numprocs
          if (myid .eq. 0)then
                  write(*,*)trim(version)
                  write(*,*)"compiler: ",compiler_version()
          endif
          call MPI_FINALIZE(ierr)
          stop
          end
    
    END
    
    cat > helloc.c << END
    #include <stdio.h>
    #include <stdlib.h>
    #include <mpi.h>
    #include <math.h>
     
    /************************************************************
    This is a simple hello world program. Each processor prints
    name, rank, and total run size.
    ************************************************************/
    int main(int argc, char **argv)
    {
        int myid,numprocs,resultlen;
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        char myname[MPI_MAX_PROCESSOR_NAME] ;
        int vlan;
        MPI_Init(&argc,&argv);
        MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
        MPI_Comm_rank(MPI_COMM_WORLD,&myid);
        MPI_Get_processor_name(myname,&resultlen); 
        printf("Hello from %s %d %d\n",myname,myid,numprocs);
        if (myid == 0 ) {
    	    MPI_Get_library_version(version, &vlan);
    	    printf("%s\n",version);
                printf("compiler: %s\n", __VERSION__);
        }
        MPI_Finalize();
    }
    END
    
    
    
    
    echo For RHEL9 Cray ftn and cc are not in out path by default
    echo so this would not work so it is commented out
    << A_COMMENT
    echo Compiling with default compilers
    cc helloc.c -o helloc.exe
    ftn hellof.f90 -o hellof.exe
    A_COMMENT
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with PrgEnv-cray
    module reset
    ml cpe-stack/25.03  PrgEnv-cray
    ml $ARCH
    ml
    
    cc helloc.c -o crayc.exe
    ftn hellof.f90 -o crayf.exe
    
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; srun -n 2 ./crayc.exe;  fi
    srun -n 2 ./crayf.exe
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with PrgEnv-gnu 
    module reset
    ml cpe-stack/25.03  PrgEnv-gnu
    ml $ARCH
    ml
    
    cc helloc.c -o gcc.exe
    ftn hellof.f90 -o gfortran.exe
    
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; srun -n 2 ./gcc.exe ; fi
    srun -n 2 ./gfortran.exe
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with PrgEnv-intel
    module reset
    ml cpe-stack/25.03  PrgEnv-intel
    ml $ARCH
    ml
    
    cc helloc.c -o intelccray.exe
    ftn hellof.f90 -o intelfcray.exe
    
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ;  srun -n 2 ./intelccray.exe ; fi
    srun -n 2 ./intelfcray.exe
    
    # as of 08/27/2026 PrgEnv-nvidia is broken on kestrel
    # so we skip this
    <<A_COMMENT
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with PrgEnv-nvidia
    module reset
    ml cpe-stack/25.03  PrgEnv-nvidia
    ml $ARCH
    ml
    
    cc helloc.c -o nvidiac.exe
    ftn hellof.f90 -o nvidiaf.exe
    
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; srun -n 2 ./nvidiac.exe ; fi
    srun -n 2 ./nvidiaf.exe
    A_COMMENT
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with IntelMPI
    echo mpiicx mpiifx intel compilers with intel mpi
    module reset
    ml oneapi/2025.3.1  intel-oneapi-mpi
    ml
    
    mpiicx helloc.c -o intelc.exe
    mpiifx hellof.f90 -o intelf.exe
    
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ;  srun  -n 2 ./intelc.exe ; fi
    srun  -n 2 ./intelf.exe
    
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with OpenMPI with gcc
    
    module reset
    ml gcc/14.2.0
    ml openmpi/5.0.3
    ml
    
    mpif90 hellof.f90 -o opengfort.exe
    mpicc  helloc.c   -o opencgcc.exe
    # as of 08/27/2026 this flag is required to run
    IRFLAG="-mca mtl ^ofi --mca btl ^ofi"
    mpirun -n 2 $IRFLAG   opengfort.exe
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; mpirun -n 2 $IRFLAG   opencgcc.exe ; fi
    
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo Compiling and running with OpenMPI with Intel backend
    
    module reset
    ml oneapi/2025.3.1
    ml openmpi/5.0.3
    ml
    
    mpif90 hellof.f90 -o openfint.exe
    mpicc  helloc.c   -o opencint.exe
    # as of 08/27/2026 this flag is required to run
    IRFLAG="-mca mtl ^ofi --mca btl ^ofi"
    mpirun -n 2 $IRFLAG   openfint.exe
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; mpirun -n 2 $IRFLAG   opencint.exe ; fi
    
    module reset
    ml gcc
    ml mpich
    ml
    
    mpif90 hellof.f90 -o mpichf.exe
    mpicc  helloc.c   -o mpichc.exe
    srun -n 2 mpichf.exe
    if [ $TF == 0 ] ; then echo ; echo RUNNING C VERSION ; srun -n 2 mpichc.exe ; fi
    
    echo ; echo ; echo +_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
    echo FOR CLEAN UP 
    echo rm crayc.exe crayf.exe gcc.exe gfortran.exe intelccray.exe intelc.exe  
    echo rm intelfcray.exe intelf.exe opencgcc.exe opencint.exe openfint.exe opengfort.exe mpichf.exe mpichc.exe
    echo rm helloc.c hellof.f90
    echo rm quick.out
    ```

