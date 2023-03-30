# Running Gaussian16 Software Jobs

*Learn about the Gaussian16 electronic structure program and how to run Gaussian16 jobs at NREL.*

!!! tip "Important"
	 To run Gaussian16, users must be a member of the Gaussian user group. To be added to the group, contact [HPC-Help](mailto:hpc-help@nrel.gov). In your email message, include your username and copy the following text agreeing not to compete with Gaussian, Inc.:
	```
	I am not actively developing applications for a competing software program, or for a project in 
	collaboration with someone who is actively developing for a competing software program. I agree 
	that Gaussian output cannot be provided to anyone actively developing for a competing software program.

	I agree to this statement.

	```

## Configuration and Default Settings

NREL currently has Gaussian16 Revision C.01 installed, and the user manual can be found at the [Gaussian website](https://gaussian.com/man).  Gaussian 16 C.01 also has an GPU version, and for instructions on how to run Gaussian 16 on GPU nodes, see [GitHub](https://github.nrel.gov/hlong/Gaussian_GPU).

Previous Gaussian 09 users sometimes may feel Gaussian 16 runs slower than Gaussian 09. That's because Gaussian G16 has changed the default accuracy into `Int=Acc2E=12 Grid=Ultrafine`, which means that individual SCF iterations will take longer with G16 than with G09. 

## Batch Submission with Use of In-Memory Filesystem (Preferred Method)

Gaussian jobs typically write large amounts of information to temporary scratch files. When many Gaussian jobs are running, this can put a large traffic load on the Lustre parallel filesystem. To reduce this load, we recommend putting the first 5 GB or so of scratch files into a local (on-node) in-memory filesystem called `/dev/shm`.

This scratch space is set automatically by the example script below. The Gaussian input file needs the following two directives to tell the program to put read-write files first in `/dev/shm` (up to 5GB below), and to put data that exceeds 5GB into files in a directory on the `/scratch` file system. An example script for batch submission is given below: 

### Sample Job Scripts

#### Eagle

Gaussian may be configured on Eagle to run on one or more physical nodes, with or without shared memory parallelism. Distributed memory, parallel setup is taken care of automatically based on settings in the SLURM script example below.

??? example "Eagle Sample Submission Script"

	```bash
	#!/bin/bash
	#SBATCH --time=48:00:00 
	#SBATCH --nodes=3
	#SBATCH --job-name=G16test
	#SBATCH --output=std.out
	#SBATCH --error=std.err
	#SBATCH --account=hpcapps
	
	# Load Gaussian module to set environment
	module load gaussian
	cd $SLURM_SUBMIT_DIR
	
	# Set script variables
	INPUT_BASENAME=G16_test
	INPUT_FILE=$INPUT_BASENAME.com
	GAUSSIAN_EXEC=g16
	MEMSIZE=5GB 
	SCRATCH=/tmp/scratch/$SLURM_JOB_ID
	SCRATCH2=/dev/shm 
	# 
	# Check on editing input file. If scratch directories 
	# are listed then file is used un-changed, if 3-line 
	# header not present, then script prepends these lines 
	# to the input file to be used in execution line 
	# 
	NUMRWFLINES=`grep "RWF" $INPUT_FILE | wc -l` 
	if [ $NUMRWFLINES -eq 1 ]; then 
	 echo "standard file found" 
	 cp $INPUT_FILE infile 
	else 
	 echo "prepending lines to input file" 
	 echo "%RWF=$SCRATCH2/,$MEMSIZE,$SCRATCH/,-1" > infile 
	 echo "%NoSave" >> infile 
	 echo " " >> infile 
	 cat $INPUT_FILE >> infile 
	fi 
	
	# 
	# Run gaussian NREL script (performs much of the Gaussian setup) 
	g16_nrel 
	
	# 
	# Set required Gaussian environment variables 
	# 
	if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then 
	 export GAUSS_LFLAGS='-vv -opt "Tsnet.Node.lindarsharg: ssh"' 
	 export GAUSS_EXEDIR=$g16root/g16/linda-exe:$GAUSS_EXEDIR 
	fi 
	export GAUSS_SCRDIR=$SCRATCH2 
	# 
	# Gaussian needs scratch directories 
	# Note: sometimes files may have been left behind in 
	# on-node memory by other jobs that terminated incorrectly 
	# so clean these to make sure there is enough space. 
	# 
	 
	mkdir $SCRATCH 
	rm $SCRATCH2/* 
	
	# Run Gaussian job 
	$GAUSSIAN_EXEC < infile >& $INPUT_BASENAME.log 
	rm infile
	
	rm $SCRATCH/*
	rmdir $SCRATCH

	```	

This script and sample Gaussian input are located at */nopt/nrel/apps/gaussian/examples*. The gaussian module is loaded by the script automatically, so the user does not need to have loaded the module before submitting the job. The g16_eagle python script edits the Default.Route file based on the SLURM environment set when the script is submitted to the queue. The user also must supply the name of the input file (`INPUT_BASENAME`). 

The user scratch space is set to a directory in the user's scratch space, with a name containing the job ID so different jobs will not overwrite the disk space. The script sets the directories for scratch files and environment variables needed by Gaussian (eg `GAUSS_SCRDIR`).

Please note that if a template input file without the header lines containing `%RWF`, and  `%NoSave` directives, the script will prepend these lines to the input file based on variables set in the script above. 

Eagle currently has 50 computing nodes with dual NVIDIA Tesla V100 GPUs and Gaussian G16 C.01 has the capability to run on those nodes using GPUs. For detailed instructions on how to run Gaussian on GPU nodes, see [GitHub](https://github.nrel.gov/hlong/Gaussian_GPU). 

To submit a job with the example script, named g16.slurm, one would type:

`sbatch g16.slurm`

#### Swift
The only difference to run on Swift is to change the scratch directory because Swift has no /tmp/scratch directory. Please change `SCRATCH=/tmp/scratch/$SLURM_JOB_ID` in the above Eagle sciprt into `SCRATCH=/scratch/$USR/$SLURM_JOB_ID`.
