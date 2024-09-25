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

NREL currently has Gaussian16 Revision C.01 installed, and the user manual can be found at the [Gaussian website](https://gaussian.com/man).  Gaussian currently doesn't have support for H100 GPUs.

Previous Gaussian 09 users sometimes may feel Gaussian 16 runs slower than Gaussian 09. That's because Gaussian G16 has changed the default accuracy into `Int=Acc2E=12 Grid=Ultrafine`, which means that individual SCF iterations will take longer with G16 than with G09. 

### Sample Job Scripts

Gaussian may be configured to run on one or more physical nodes, with or without shared memory parallelism. Distributed memory, parallel setup is taken care of automatically based on settings in the SLURM script example below, which should work on Eagle, Swift, and Kestrel.

??? example "Sample Submission Script"

	```bash
	#!/bin/bash
	#SBATCH --job-name G16_test
	#SBATCH --nodes=2
	#SBATCH --time=1:00:00
	#SBATCH --account=[your account]
	#SBATCH --error=std.err
	#SBATCH --output=std.out
	#SBATCH --exclusive
	#SBATCH -p debug
	
	# Load Gaussian module to set environment
	module load gaussian python
	module list
	
	cd $SLURM_SUBMIT_DIR
	
	INPUT_BASENAME=G16_test
	GAUSSIAN_EXEC=g16
	
	if [ -e /dev/nvme0n1 ]; then
	SCRATCH=$TMPDIR
	echo "This node has a local storage and will use $SCRATCH as the scratch path"
	else
	SCRATCH=/scratch/$USER/$SLURM_JOB_ID
	echo "This node does not have a local storage drive and will use $SCRATCH as the scratch path"
	fi
	
	mkdir -p $SCRATCH
	
	export GAUSS_SCRDIR=$SCRATCH
	
	# Run gaussian NREL script (performs much of the Gaussian setup)
	g16_nrel
	
	#Setup Linda parameters
	if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then 
	export GAUSS_LFLAGS='-vv -opt "Tsnet.Node.lindarsharg: ssh"' 
	export GAUSS_EXEDIR=$g16root/g16/linda-exe:$GAUSS_EXEDIR 
	fi 
	
	# Run Gaussian job 
	$GAUSSIAN_EXEC < $INPUT_BASENAME.com >& $INPUT_BASENAME.log 
	
	rm $SCRATCH/*
	rmdir $SCRATCH
	```

This script and sample Gaussian input are located at */nopt/nrel/apps/gaussian/examples*. The gaussian module is loaded by the script automatically, so the user does not need to have loaded the module before submitting the job. The g16_nrel python script edits the Default.Route file based on the SLURM environment set when the script is submitted to the queue. The user also must supply the name of the input file (`INPUT_BASENAME`). 

The user scratch space is set to a directory in the default scratch space, with a name containing the job ID so different jobs will not overwrite the disk space. The default scratch space is /tmp/scratch when a local disk is available or /scratch/$USER. The script sets the directories for scratch files and environment variables needed by Gaussian (eg `GAUSS_SCRDIR`).

To submit a job with the example script, named g16.slurm, one would type:

`sbatch g16.slurm`

