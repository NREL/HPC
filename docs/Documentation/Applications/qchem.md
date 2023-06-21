# Using Q-Chem

*Q-Chem is a comprehensive *ab initio* quantum chemistry package with special strengths in excited state methods, non-adiabatic coupling, solvation models, explicitly correlated wave-function methods, and cutting-edge density functional theory (DFT).* 

## Running Q-Chem

The `q-chem` module should be loaded to set up the necessary environment. The `module help` output can provide more detail. In particular, the modulefile does not set the needed environment variable `QCSCRATCH`, as this is likely unique for each run. `QCLOCALSCR` is set by default to `/tmp/scratch`, but one may wish to point to a more persistent location if files written to local scratch need to be accessed after the job completes. Users can easily do this in their Slurm scripts or at the command line via `export` (Bash) or `setenv` (csh). 

The simplest means of starting a Q-Chem job is via the supplied `qchem` wrapper. The general syntax is: 

`qchem -slurm <-nt number_of_OpenMP_threads> <input file> <output file> <savename>`

For example, to run a job with 36 threads:

`qchem -slurm -nt 36 example.in`

!!! tip "Note"
	 The Q-Chem input file must be in the same directory in which you issue the qchem command. In other words, `qchem ... SOMEPATH/<input file>` won't work. 

For a full list of which types of calculation are parallelized and the types of parallelism, see the [Q-Chem User's Manual](https://manual.q-chem.com/5.3/).

To save certain intermediate files for, *e.g.*, restart, a directory name needs to be provided. If not provided, all scratch files will be automatically deleted at job's end by default. If provided, a directory `$QCSCRATCH/savename` will be created and will hold saved files. In order to save all intermediate files, you can add the `-save` option. 

A template Slurm script to run Q-Chem with 36 threads is:

??? example "Sample Submission Script"

	```bash
	#SBATCH --job-name=my_qchem_job
	#SBATCH --account=my_allocation_ID
	#SBATCH --ntasks=36
	#SBATCH --time=01:00:00
	#SBATCH --mail-type=BEGIN,END,FAIL
	#SBATCH --mail-user=your_email@domain.name
	#SBATCH --output=std-%j.out
	#SBATCH --error=std-%j.err
	 
	# Load the Q-Chem environment
	module load q-chem
	 
	# Go to the location of job files, presumably from where this file was submitted
	cd $SLURM_SUBMIT_DIR
	 
	# Set up scratch space
	SCRATCHY=/scratch/$USER/${SLURM_JOB_NAME:?}
	if [ -d $SCRATCHY ]
	then
	   rm -r $SCRATCHY
	fi
	mkdir -p $SCRATCHY
	export QCSCRATCH=$SCRATCHY
	 
	# Move files over
	cp * $SCRATCHY/.
	cd $SCRATCHY
	 
	# Start run. Keep restart files without intermediate temp files in directory called "my_save"
	qchem -nt 36 job.in job.out my_save
	```

To run this script on Swift, the number of threads can be changed to 64. 

A large number of example Q-Chem input examples are available in `/nopt/nrel/apps/q-chem/<version>/samples`.

