# Using Q-Chem

*Q-Chem is a comprehensive *ab initio* quantum chemistry package with special strengths in excited state methods, non-adiabatic coupling, solvation models, explicitly correlated wave-function methods, and cutting-edge density functional theory (DFT).* 

## Running Q-Chem

The `q-chem` module should be loaded to set up the necessary environment. The `module help` output can provide more detail. In particular, the modulefile does not set the needed environment variable `QCSCRATCH`, as this is likely unique for each run. Users should do this in their Slurm scripts or at the command line via `export` (bash) or `setenv` (csh). 

The simplest means of starting a Q-Chem job is via the supplied `qchem` wrapper. The general syntax is: 

`qchem -slurm <-nt number_of_OpenMP_threads> <input file> <output file> <savename>`

For example, to run a job with 104 threads:

`qchem -slurm -nt 104 example.in`

!!! tip "Note"
	 The Q-Chem input file must be in the same directory in which you issue the qchem command. In other words, `qchem ... SOMEPATH/<input file>` won't work. 

For a full list of which types of calculation are parallelized and the types of parallelism, see the [Q-Chem User's Manual](https://manual.q-chem.com/6.2/).

To save certain intermediate files for, *e.g.*, restart, a directory name needs to be provided. If not provided, all scratch files will be automatically deleted at job's end by default. If provided, a directory `$QCSCRATCH/savename` will be created and will hold saved files. In order to save all intermediate files, you can add the `-save` option. 

A template Slurm script to run Q-Chem with 104 threads is:

### Sample Submission Script for Kestrel

```
 	#!/bin/bash
	#SBATCH --job-name=my_qchem_job
	#SBATCH --account=[my_allocation_ID]
	#SBATCH --nodes=1
 	#SBATCH --tasks-per-node=104
  	#SBATCH --time=01:00:00
   	#SBATCH --exclusive
	#SBATCH --mail-type=BEGIN,END,FAIL
	#SBATCH --mail-user=your_email@domain.name
	#SBATCH --output=std-%j.out
	#SBATCH --error=std-%j.err
	 
	# Load the Q-Chem environment
	module load q-chem/6.2

	if [ -e /dev/nvme0n1 ]; then
	 SCRATCH=$TMPDIR
	 echo "This node has a local storage and will use $SCRATCH as the scratch path"
	else
	 SCRATCH=/scratch/$USER/$SLURM_JOB_ID
	 echo "This node does not have a local storage drive and will use $SCRATCH as the scratch path"
	fi

	mkdir -p $SCRATCH

	export QCSCRATCH=$SCRATCH
	export QCLOCALSCR=$SCRATCH

	jobnm=qchem_test

	if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then
	 QCHEMOPT="-mpi -np $SLURM_NTASKS"
	else
	 QCHEMOPT="-nt $SLURM_NTASKS"
	fi

	echo Running Q-Chem with this command: qchem $QCHEMOPT $jobnm.com $jobnm.out
	qchem $QCHEMOPT $jobnm.com $jobnm.out

	rm $SCRATCH/*
	rmdir $SCRATCH
```

To run this script on systems other than Kestrel, the number of threads should be changed accordingly. 

A large number of example Q-Chem input examples are available in `/nopt/nrel/apps/q-chem/<version>/samples`.

## Running BrianQC
BrianQC is the GPU version of Q-Chem and can perform Q-Chem calculations on GPUs, which is significantly faster for some larger ab initio jobs. BrianQC uses the same input file as Q-Chem. Below is a sample slurm script for BrianQC, which should be submitted on the GPU login nodes of Kestrel. If running on Swift, please also add "#SBATCH -p gpu" to the header of this script.
```
 	#!/bin/bash
	#SBATCH --job-name=my_qchem_job
	#SBATCH --account=[my_allocation_ID]
	#SBATCH --nodes=1
 	#SBATCH --tasks-per-node=104
  	#SBATCH --time=01:00:00
   	#SBATCH --gres=gpu:[number of gpu]
	#SBATCH --mem=[requested memory]
	#SBATCH --mail-type=BEGIN,END,FAIL
	#SBATCH --mail-user=your_email@domain.name
	#SBATCH --output=std-%j.out
	#SBATCH --error=std-%j.err
	 
	# Load the Q-Chem environment
	module load brianqc

	if [ -e /dev/nvme0n1 ]; then
	 SCRATCH=$TMPDIR
	 echo "This node has a local storage and will use $SCRATCH as the scratch path"
	else
	 SCRATCH=/scratch/$USER/$SLURM_JOB_ID
	 echo "This node does not have a local storage drive and will use $SCRATCH as the scratch path"
	fi

	mkdir -p $SCRATCH

	export QCSCRATCH=$SCRATCH
	export QCLOCALSCR=$SCRATCH

	jobnm=qchem_test

	if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then
	 QCHEMOPT="-gpu -mpi -np $SLURM_NTASKS"
	else
	 QCHEMOPT="-gpu -nt $SLURM_NTASKS"
	fi

	echo Running Q-Chem with this command: qchem $QCHEMOPT $jobnm.com $jobnm.out
	qchem $QCHEMOPT $jobnm.com $jobnm.out

	rm $SCRATCH/*
	rmdir $SCRATCH
```


   
