# Sample Slurm Batch Scripts

Thes scripts showcase various program flow techniques by leveraging bash and slurm features.  For 
MPI examples we assume we will be using mpt MPI but the scripts will work with Intel 
also.  

These examples run quickly and most can be run in the debug partition.  For most 
scripts the submission line is of the form:

```
sbatch –A myaccount --partition=debug --nodes=N script-to-run
```

See testit for more details.


* [`hostname.sh`](./hostname.sh) - Simple script that just does hostname on all cores.  

* [`multinode-task-per-core.sh`](./multinode-task-per-core.sh) - Example of mapping process execution to each core on an arbitrary amount of nodes.

* [`openmp.sh`](./openmp.sh) - Runs an OpenMP  program with a set number of cores per node.

* [`simple.sh`](./simple.sh) - Runs an MPI program with a set number of cores per node.

* [`hybrid.sh`](./hybrid.sh) - Runs an MPI/OpenMP program with a set number of cores per node.

* [`affinity.sh`](./affinity.sh) - Runs an MPI/OpenMP program with a set number of cores per node, looking at various affinity settings.

* [`newdir.sh`](./newdir.sh) - Create a new directory for each run, save script, environment, and output

* [`fromenv.sh`](./fromenv.sh) - Get input filename from your environment.

* [`mpmd.sh`](./mpmd.sh) - MPI with different programs on various cores - two methods.

* [`mpmd2.sh`](./mpmd2.sh) - MPI with different programs on two nodes with different counts on each node.

* [`multi.sh`](./multi.sh) - Multiple applications in a single sbatch submission.

* [`gpucpu.sh`](./gpucpu.sh) - Run a cpu and gpu job in the same script concurrently.

* [`FAN.sh`](./FAN.sh) - A bash script, not a slurm script for submitting a number of jobs with dependencies.  

* [`CHAIN.sh`](./CHAIN.sh) - A bash script, not a slurm script for submitting a number of jobs with dependencies.  A simplified version of FAN, only submits 5 jobs.

* [`old_new_.sh`](./old_new_.sh) - Job submitted by FAN.sh or CHAIN.sh.  Can copy old run data to new directories and rerun.  

* [`uselist.sh`](./uselist.sh) - Array jobs, multiple jobs submitted with a single script. "Use the slurm option --array=1-24 to submit"

* [`redirect.sh`](./redirect.sh) - Low level file redirection, allows putting slurm std{err,out} anywhere.

* [`multimax.sh`](./multimax.sh) - Multiple nodes, multiple jobs concurrently with also forcing affinity.

* [`local.sh`](./local.sh) - slurm script showing how to use local \"tmp\" disk.

* [`testit`](./testit) - A bash script for running all of the examples.


## Source code, extra scripts, and makefile to use with the above scripts.
### Note:

```
These files are in a subdirectory for organizational purposes.  After checkout, go 
to this directory and do a make install which will compile and copy files up one level.  
Also, you can create a python environment for the examples by sourcing the file jupyter.sh.
 
```

* [`phostone.c`](source/phostone.c) - Glorified hello world in hybrid MPI/OpenMP for some examples.

* [`doarray.py`](source/doarray.py) - Wrapper for uselist.sh, creates inputs and runs uselist.sh.

* [`stf_01.f90`](source/stf_01.f90) - Simple finite difference code to run as an example.

* [`c_ex02.c`](source/c_ex02.c0) - Simple example in C.

* [`f_ex02.f90`](source/f_ex02.f90) - Same as c_ex02.c but in Fortran.

* [`makefile`](source/makefile) - Makefile for examples. Loads MPT module then compiles.

* [`makefile.intel`](source/makefile) - Makefile for examples. Loads INTEL MPI module then compiles.

* [`hymain.c`](source/hymain.c) - MPI program that calls a routine that uses GPUs.

* [`hysub.cu`](source/hysub.cu) - Simple routine that accesses GPUs. 

* [`invertc.c`](source/invertc.c) - Matrix inversion program with a thread assigned to each of 4 inverts.

* [`slowinvert.f90`](source/slowinvert.f90) - Matrix inversion program with a single thread assigned to each of many inverts.

* [`logfile`](source/logfile) - Example output from running testit.

* [`report.py`](source/report.py) - A python mpi4py program for showing mapping tasks to cores.

* [`prolog.py`](source/prolog.py) - A python script designed to be run as a srun prolog command.

* [`spam.c`](source/spam.c) - Source for a C/python library/module for mapping tasks to cores.

* [`setup.py`](source/setup.py) - Build file for spam.c. See multimax.sh.

* [`jupyter.sh`](source/jupyter.sh) - Create jupyter/mpi4py/pandas environment with a user defined version of MPI. 

* [`tunnel.sh`](source/tunnel.sh) - Bash function for creating a ssh tunnel to connect to a jupyter notebook.  
 
* [`tymer`](source/tymer) - Glorified wall clock timer.

* [`slurm_variables`](source/slurm_variables) - List of slurm variables.


### Remove block comments
```
You can get copies of the scripts without block comments by running the command:

for script in `ls *sh` ; do
    out=`echo $script | sed s/.sh$/.slurm/`
    echo $out
    sed  '/:<<++++/,/^++++/d' $script > $out
    chmod 750 $out
done
```

### Intel MPI

```
These scripts are set up to run using mpt MPI.  You can create scripts
for using Intel MPI by running the following commands in the slurm directory.

for script in `ls *sh` ; do
    out=`echo $script | sed s/.sh$/.impi/`
    echo $out
    sed  's,module load mpt,module load intel-mpi/2020.1.217,' $script > $out
    chmod 750 $out
done

Change old_new.sh to old_new.intel in FAN.intel and CHAIN.intel.

Make the examples with Intel MPI by running make makefile.intel in the source directory.

To build the conda enviroment with Intel MPI change the module load command in source/jupyter.sh.

```
