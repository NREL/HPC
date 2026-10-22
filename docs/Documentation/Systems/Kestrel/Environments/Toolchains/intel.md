# Compile and run: *Intel oneAPI compilers & MPI/OpenMP*

### Introduction
This page shows how to compile and run programs using Intel's oneAPI tool chain.  We'll look at building using their MPI and Fortran and C compilers.  We'll also use Intel's MPI libraries but actually compile using `gfortran` and `gcc`. We also have a makefile for building with Cray MPI and Intel backend compilers.

Our example programs are hybrid MPI/Openmp so we'll show commands for building hybrid programs.  If your program is pure MPI the only change you need to make to the build process is to remove the compile line option `-fopenmp`.  

Sample makefile, source codes, and runscript for on Kestrel can be found in our [Kestrel Repo](https://github.com/NatLabRockies/HPC/tree/master/kestrel)  under the Toolchains folder.  There are individual directories for source, makefiles, and scripts or you can download the intel.tgz file containing all required files.  The source differs slightly from what is shown here.  There is an extra file *`triad.c`* that gets compiled along with the Fortran and C programs discussed below.  This file does some "dummy" work to allow the programs to run for a few seconds.  


### module loads for compile

These are the module you will need for compiles:

```
module load oneapi 
module load intel-oneapi-mpi        
```


### module loads for run
You should load the modules:

```
module load oneapi           
module load intel-oneapi-mpi 
```
This will ensure you have access to all of the required libraries and the  specific variables required to run on Kestrel are set.  


### Building programs

As discussed above we can build with Intel (`ifx`, `icx`) or GNU (`gcc`, `gfortran`) underlying compilers.  The four options are:

1. Fortran with: Intel MPI and Intel Fortran compiler (`ifx`)
2. C with: Intel MPI and Intel C compiler compiler (`icx`)
3. Fortran with: Intel MPI with `gfortran` Fortran compiler
4. C with: Intel MPI with `gcc` C compiler

Here's what the compile lines should be where we add the -qopenmp option for Openmp and the optimization flag `-O3`.

#### 1. Fortran with: Intel MPI and Intel Fortran compiler

```
mpiifx -O3 -g -qopenmp  ex1.f90  
```

#### 2. C with: Intel MPI and Intel C compiler 
```
mpiicx -O3 -g -fopenmp  ex1.c  -o ex_c
```

Note that the commands `mpicc` and `mpif90` actually call `gcc` and `gfortran` instead of the Intel compilers.  

#### 3. Fortran with: Intel MPI with gfortran Fortran compiler

```
mpif90 -O3 -g -fopenmp  ex1.f90 
```
#### 4. C with: Intel MPI with gcc C compiler
```
mpicc -O3 -g -fopenmp  ex1.f90 
```


### Example programs

We have two example MPI/OpenMP programs, `ex1.c` and `ex1.f90`.  They are more or less identical in function.  They first print MPI Library and compiler information.  For example the fortran example compiled with `mpiifort` reports:

```
  Fortran MPI TASKS            4
 Intel(R) MPI Library 2021.8 for Linux* OS

 Intel(R) Fortran Intel(R) 64 Compiler Classic for applications running on Intel
```

For mpif90 we get:

```
  Fortran MPI TASKS            4
 Intel(R) MPI Library 2021.8 for Linux* OS

 GCC version 13.1.0
```

Note in these cases we have the same MPI library but different compilers.

The programs call a routine, *triad*. It keeps the cores busy for about 4 seconds.  This allows the OS to settle down.  Then for each MPI task and each openmp thread we get a line of the form:

```
task 0001 is running on x9000c3s2b0n0 thread=   2 of   3 is on core  054
```

This is saying that MPI task 1 is running on node x9000c3s2b0n0.  The task has 3 openmp threads and the second is running on core 54.




### Example makefile

The `triad.c` file contains the routines that keeps the cores busy for 4 seconds.  This is common to both the fortran and C versions of our codes. As discussed above our main codes are ex1.c and ex1.f90.  Our makefile will build for:

#### 1. Fortran with: Intel MPI and Intel Fortran compiler
#### 2. C with: Intel MPI and Intel C compiler
#### 3. Fortran with: Intel MPI with gfortran Fortran compiler
#### 4. C with: Intel MPI with gcc C compiler


The makefile has an interesting "trick".  The default target is recurse.  This target loads the modules then calls make again using the same makefile but with the targets  intel and gnu.  By using this "trick" you don't have to load modules before the make.  

The targets intel and gnu each have a dependency to compile triad with either Intel or gcc compilers.  Then the final applications are built with Intel MPI and again the either Intel or gnu.

The final MPI codes are: 

* gex_c : `gcc`
* gex_f : `gfortran`
* ex_c  : Intel C (`icx`)
* ex_f  : Intel Fortran (`ifort`)


### Run script

1. Makes a new directory, copies the required files and goes there
2. Does a make with output going into make.log
3. Sets the number of MPI tasks and openmp threads
4. Sets some environmental variables to control and report on threads (discussed below)
5. module commands
	1. module reset          
	2. module load oneapi           
	3. module load intel-oneapi-mpi
6. Creates a string with all of our srun options (discussed below)
7. Calls srun on each version of our program
	1. output goes to *.out
	2. Report on thread placement goes to *.info

Our script sets these openmp related variables.  The first is familiar. KMP_AFFINITY is unique to Intel compilers.  In this case we are telling the OS to scatter (spread) out our threads.  OMP_PROC_BIND=spread does the same thing but it is not unique to Intel compilers. So in this case KMP_AFFINITY is actually redundent.  

```
export OMP_NUM_THREADS=3
export KMP_AFFINITY=scatter
export OMP_PROC_BIND=spread
```

The next line 

```
export BIND="--cpu-bind=v,cores"
```

is not technically used as an environmental variable but it will be used to create the srun command line.  Passing `--cpu-bind=v` to `srun` will cause it to report threading information.  The `cores` option tells `srun` to "Automatically generate masks binding tasks to cores."  There are many other binding options as described in the `srun` man page. This setting works well for many programs.


Our `srun` command line options for 2 tasks per node and 3 threads per task are:

```
--mpi=pmi2 --cpu-bind=v,cores --threads-per-core=1 --tasks-per-node=2 --cpus-per-task=3
```

* `--mpi=pmi2` : tells srun to use a particular launcher (This is optional.)
* `--cpu-bind=v,cores` : discussed above
* `--threads-per-core=1` : don't allow multiple threads to run on the same core.  Without this option it is possible for multiple threads to end up on the same core, decreasing performance.  
* `--cpus-per-task=3` : The `cpus-per-task` should always be equal to `OMP_NUM_THREADS`.


The final thing the script does is produce a results report.  This is just a list of mapping of mpi tasks and threads.  There should not be any repeats in the list.  There will be "repeats" of cores but on different nodes.   There will be "repeats" of nodes but with different cores.

You can change the values for `--cpu-bind`, `OMP_PROC_BIND`, and `threads-per-core` to see if this list changes.

	
	
??? example "triad.c"

	```c    	
	#include <stdio.h>
	#include <mpi.h>
	#include <omp.h>
	#include <unistd.h>
	int sched_getcpu();
	double dotriad(int *myid);
	/************************************************************
	This is a simple hybrid hello world program.
	Prints MPI information 
	For each task/thread
	  task id
	  node name for task
	  thread id
	  # of threads for the task
	  core on which the thread is running
	************************************************************/
	int main(int argc, char **argv)
	{
		int myid,numtasks,resultlen;
		int did;
		char version[MPI_MAX_LIBRARY_VERSION_STRING];
		char myname[MPI_MAX_PROCESSOR_NAME] ;
		int vlan;
		int mycore;
		double wait;
		MPI_Init(&argc,&argv);
		MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
		MPI_Comm_rank(MPI_COMM_WORLD,&myid);
		MPI_Get_processor_name(myname,&resultlen); 
		if (myid == 0 ) {
			printf(" C MPI TASKS %d\n",numtasks);
			MPI_Get_library_version(version, &vlan);
				printf("MPI VERSION: %s\n",version);
				printf("BACKEND VERSION: %s\n",__VERSION__);
		}
	// dotriad runs "triad", in parallel, for 4 seconds to give threads time to settle
	//  if input to triad is negative run for -# seconds
	//  if >=0 run "triad", in parallel one more time and give report to stderr
		did=-4;
		wait=dotriad(&did);
	#pragma omp parallel 
	  {
	#pragma omp critical
		mycore=sched_getcpu();
		printf(" task %04d is running on %s  thread %3d of %3d is on core %03d\n",
				myid,
				myname,
				omp_get_thread_num(),
				omp_get_thread_num(),
				mycore);
	  }
	  if (myid == 0)printf("ran triad for %10.2f seconds\n",wait);
	// run "triad", in parallel one more time and give report to stderr
		did=myid;
	//    wait=dotriad(&did);
		MPI_Finalize();
	}
	```
	

	
??? example "ex1.f90"
	```fortran 
	
	! This is a simple hybrid hello world program.
	! Prints MPI information
	! For each task/thread
	!  task id
	!  node name for task
	!  thread id
	!  # of threads for the task
	!  core on which the thread is running
	
	module getit
	contains
	!! Get the core on which a thread is running
	  function get_core_c()
		  USE ISO_C_BINDING, ONLY: c_long, c_char, C_NULL_CHAR, c_int
		  implicit none
		  integer, parameter :: in8 = selected_int_kind(12)
		  integer(in8) :: get_core_c
		  interface
			 integer(c_long) function cfunc() BIND(C, NAME='sched_getcpu')
				USE ISO_C_BINDING, ONLY: c_long, c_char
			 end function cfunc
		  end interface
		  get_core_c = cfunc()
	   end function
	!! runtriad runs "triad", in parallel, for 4 seconds to give threads time to settle
	   function runtriad(myin)
		  USE ISO_C_BINDING, ONLY: c_double,c_int
		  implicit none
		  double precision :: runtriad
		  integer myin
		  integer(c_int) :: myid
	
		  interface
				real(c_double) function cfunc(myid) BIND(C, NAME='dotriad')
				USE ISO_C_BINDING, ONLY: c_double,c_int
				integer(c_int)  :: myid
			 end function cfunc
		  end interface
		  myid=myin
		  runtriad = cfunc(myin)
	   end function
	end module
	
	program hybrid
		use getit
		use ISO_FORTRAN_ENV
		implicit none
		include 'mpif.h'
		integer numtasks,myid,ierr
		character (len=MPI_MAX_PROCESSOR_NAME):: myname
		character(len=MPI_MAX_LIBRARY_VERSION_STRING+1) :: version
		integer mylen,vlan,mycore,tin
		double precision wait
		integer OMP_GET_MAX_THREADS,OMP_GET_THREAD_NUM
		call MPI_INIT( ierr )
		call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
		call MPI_COMM_SIZE( MPI_COMM_WORLD, numtasks, ierr )
		call MPI_Get_processor_name(myname,mylen,ierr)
	! print the MPI libraty version
		if (myid .eq. 0)then
		  write(*,*)"Fortran MPI TASKS ",numtasks
		  call MPI_Get_library_version(version, vlan, ierr)
		  write(*,*)"MPI VERSION: ",trim(version)
		  write(*,*)"BACKEND COMPILER: ",trim(ADJUSTL(COMPILER_VERSION()))
		endif
	!! runtriad runs "triad", in parallel, for 4 seconds to give threads time to settle
	!!  if input to triad is negative run for -# seconds
	!!  if >=0 run "triad", in parallel one more time and give report to stderr
	
		tin=-4
		wait=runtriad(tin)
	!$OMP PARALLEL
	!$OMP CRITICAL
		mycore=get_core_c()
		write(unit=*,fmt="(a,i4.4,a,a)",advance="no") &
					" task ",myid, " is running on ",trim(myname)
		write(unit=*,fmt="(a,i3,a,i3,a,1x,i3.3)") &
				" thread= ",OMP_GET_THREAD_NUM(), &
				" of ",OMP_GET_MAX_THREADS(),     &
				" is on core ",mycore
	!$OMP END CRITICAL
	!$OMP END PARALLEL
		if (myid .eq. 0) write(*,fmt="(a,f10.2,a)")" ran triad for ",wait," seconds"
	!! run "triad", in parallel one more time and give report to stderr
	!!    wait=runtriad(myid)
		call MPI_FINALIZE(ierr)
	end program
	```
	
??? example "ex1.c"

	```c
	#include <stdio.h>
	#include <mpi.h>
	#include <omp.h>
	#include <unistd.h>
	int sched_getcpu();
	double dotriad(int *myid);
	/************************************************************
	This is a simple hybrid hello world program.
	Prints MPI information 
	For each task/thread
	  task id
	  node name for task
	  thread id
	  # of threads for the task
	  core on which the thread is running
	************************************************************/
	int main(int argc, char **argv)
	{
		int myid,numtasks,resultlen;
		int did;
		char version[MPI_MAX_LIBRARY_VERSION_STRING];
		char myname[MPI_MAX_PROCESSOR_NAME] ;
		int vlan;
		int mycore;
		double wait;
		MPI_Init(&argc,&argv);
		MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
		MPI_Comm_rank(MPI_COMM_WORLD,&myid);
		MPI_Get_processor_name(myname,&resultlen); 
		if (myid == 0 ) {
			printf(" C MPI TASKS %d\n",numtasks);
			MPI_Get_library_version(version, &vlan);
				printf("MPI VERSION: %s\n",version);
				printf("BACKEND VERSION: %s\n",__VERSION__);
		}
	// dotriad runs "triad", in parallel, for 4 seconds to give threads time to settle
	//  if input to triad is negative run for -# seconds
	//  if >=0 run "triad", in parallel one more time and give report to stderr
		did=-4;
		wait=dotriad(&did);
	#pragma omp parallel 
	  {
	#pragma omp critical
		mycore=sched_getcpu();
		printf(" task %04d is running on %s  thread %3d of %3d is on core %03d\n",
				myid,
				myname,
				omp_get_thread_num(),
				omp_get_thread_num(),
				mycore);
	  }
	  if (myid == 0)printf("ran triad for %10.2f seconds\n",wait);
	// run "triad", in parallel one more time and give report to stderr
		did=myid;
	//    wait=dotriad(&did);
		MPI_Finalize();
	}
    ```
??? example "makefile"
	```makefile    
	SHELL:=/usr/bin/bash
	
	recurse:
		module reset                        ; \
		module load  oneapi                 ; \
		module load intel-oneapi-mpi        ; \
		$(MAKE) -f $(firstword $(MAKEFILE_LIST)) intel gnu rmo
	
	intel: ex_f ex_c 
	gnu: gex_f gex_c 
	
	#Intel mpi  with intel compilers
	PF90=mpiifx
	PCC=mpiicx
	ICC=icx
	
	
	#Intel mpi with gcc and gfortran
	PG90=mpifc
	PGCC=mpicc
	
	OPS_I=-O3 -g -qopenmp
	OPS_G=-O3 -g -fopenmp
	
	
	triadi.o: `triad.c`
		$(ICC) $(OPS_I) $(XOPT) -c `triad.c` -o triadi.o
	
	triadg.o: `triad.c`
		gcc $(OPS_G) -c `triad.c` -o triadg.o
		
	
	ex_f: ex1.f90 triadi.o
		$(PF90) $(OPS_I)       ex1.f90  triadi.o -o ex_f
	
	ex_c: export I_MPI_CC=$(ICC)
	ex_c: ex1.c 
		$(PCC) $(OPS_I) $(XOPT)  ex1.c  triadi.o -o ex_c
	
	gex_f: ex1.f90 triadg.o
		$(PG90) $(OPS_G)  ex1.f90 triadg.o  -o gex_f
	
	gex_c: ex1.c triadg.o
		$(PGCC) $(OPS_G)       ex1.c  triadg.o -o gex_c
	
	
	clean:rmo
		rm -rf  ex_f ex_c gex_f gex_c intel.tar exc exf
	
	#clean up after the compile
	rmo:
		rm -rf *o *mod*
	
	tar:
		tar -cf intel.tar ex1.c ex1.f90 makefile script `triad.c` makecray
	```	

??? example "script"
    ```bash
      
    #!/usr/bin/bash
    #SBATCH --job-name="run_intel_mpi"
    #SBATCH --nodes=2
    #SBATCH --exclusive
    #SBATCH --export=ALL
    #SBATCH --time=00:02:00
    #SBATCH --partition=debug

    BASE=`pwd`

    #Make a new directory
    mkdir $SLURM_JOB_ID
    cd $SLURM_JOB_ID

    #Copy everything 
    cat $0 > script
    cp $BASE/{ex1.c,ex1.f90,`triad.c`,makefile,makecray,script} .

    #Build our programs. Put make output in make.log
    make  > make.log 2>&1

    #These lines will use every core on a node
    # tasks per node
      export NTPN=13
    # threads per task
      export OMP_NUM_THREADS=`echo "104 / $NTPN" | bc `

    #These lines will use a subets of the cores
    # tasks per node
      export NTPN=2
    # threads per task
      export OMP_NUM_THREADS=3

      export OMP_PROC_BIND=spread
      export BIND="--cpu-bind=v,cores"

    module reset                  
    module load oneapi           
    module load intel-oneapi-mpi  

    #Show what our srun command line is
    export SRUNOPTS=`echo --mpi=pmi2 $BIND  --threads-per-core=1 --tasks-per-node=$NTPN --cpus-per-task=$OMP_NUM_THREADS`
    printenv SRUNOPTS  > srunline
    # The *.out files contain normal progam output
    # The *.info files contain thread mapping info
        srun $SRUNOPTS  ./ex_c > ex_c.out 2> ex_c.info
        srun $SRUNOPTS  ./ex_f > ex_f.out 2> ex_f.info

        srun $SRUNOPTS  ./gex_c > gex_c.out 2> gex_c.info
        srun $SRUNOPTS  ./gex_f > gex_f.out 2> gex_f.info

    #optional run PrgEnv-intel
    #make -f makecray

    #Summerize the results
    #This line pulls the node name and core for each MPI task/OpenMP thread.  There sould not be any duplicates.
    #There will be "repeats" of cores but on different nodes.  
    #There will be "repeats" of nodes but with different cores.
    for f in *out ; do echo $f ; grep task $f |  awk '{print $6,$14}' | sort -k2,2 ; done > results   

    mv ../slurm-$SLURM_JOB_ID.out .
    ```
  
"Bonus" makefile for building/runing with Cray MPI and Intel compilers.	

??? example "makecray"

	```makefile   
	 
	SHELL:=/usr/bin/bash
	
	recurse:
		module reset                ; \
		module load cpe-stack       ; \
		module load PrgEnv-intel    ; \
		module load craype-x86-spr  ; \
		export OMPF=-qopenmp        ; \
		$(MAKE) -f $(firstword $(MAKEFILE_LIST)) run
	
	run: both srun
	
	both: exc exf tmps
	
	exf: ex1.f90 triad.o
	
	exc: ex1.c triad.o
	
	triad.o: `triad.c`
		cc  -g -O3 $(OMPF) -c `triad.c`
	
	exf:
		ftn -g -O3 $(OMPF)  ex1.f90 triad.o -o exf
	
	exc:
		cc  -g -O3 $(OMPF) ex1.c triad.o -o exc
	
	clean:
		rm -f triad.o exf exc *mod exfcray.out exccray.out
	
	tmps: 
		rm -f *o *mod
	
	srun:
		export OMP_NUM_THREADS=3 ; srun --threads-per-core=1 --tasks-per-node=2 --cpus-per-task=3  ./exf > exfcray.out
		export OMP_NUM_THREADS=3 ; srun --threads-per-core=1 --tasks-per-node=2 --cpus-per-task=3  ./exc > exccray.out
	```
	
