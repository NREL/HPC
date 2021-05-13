#!/bin/bash 
#SBATCH --job-name="array_job"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
###### sending output to /dev/null will suppress it
###### this is a good idea for array jobs lest it
###### create extra output for each subjob
#SBATCH -o /dev/null
#SBATCH --exclusive=user
#SBATCH --mem=50000

:<<++++

Author: Tim Kaiser

This script is designed to run array jobs.  Array jobs are often
a collection of similar jobs with different inputs.  


When an slurm runs a collection of array jobs it assigns two additional
variables:

SLURM_ARRAY_JOB_ID
SLURM_ARRAY_TASK_ID

Here we use SLURM_ARRAY_TASK_ID (renamed SUB_ID) to select a line from
an input file in_list.  It is expected that in_list has a line for each
SLURM_ARRAY_TASK_ID. Thus we can run N instacance of our program with
different inputs.  A typical invocation would be

sbatch -A account --array=1-24 uselist.sh

SLURM_ARRAY_TASK_ID would be in the range 1-24 and each instance would
grab one of the 24 lines of our input file.  

The script creates a top level directory for all jobs and then a subdirectroy
again 1-24 for each subjob.  The subdirectory contains a copy of the input,
output, the node on which it was run, our script, the environment, and timing
information.  

Here we run program that does four matrix inversions.  The matricies 
are set up based on the command line input.  For example

./invertc 10 56 23 43 400

where 400 is the size and the other integers are used to set values
for the matricies.

As a convenience we have the  python script doarray.py that creates the 
input file in_file and then runs this script.  doarray.py takes an account
string as input.

Or, you can get a version of in_list from source/in_list and run directly.

++++


# example invocation
# sbatch -A account --array=1-24 uselist.sh



# conda is needed to get a recent version of python
module load conda

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=2


#run with either the C or python version of our program
#export EXE=invertp.py
export EXE=invertc


# go to our starting directory
cd $SLURM_SUBMIT_DIR

# get the JOB and SUBJOB ID
if [[ $SLURM_ARRAY_JOB_ID ]] ; then
	export JOB_ID=$SLURM_ARRAY_JOB_ID
	export SUB_ID=$SLURM_ARRAY_TASK_ID
else
	export JOB_ID=$SLURM_JOB_ID
	export SUB_ID=1
fi

# make a top level directory for the job 
# if it does not already exist
mkdir -p $JOB_ID
cd $JOB_ID


# make a directory for the subjob and go there
mkdir -p $SUB_ID
cd $SUB_ID
# Make a copy of our script
cat $0 > myscript



# Get the name of our LIST, default to in_list
if [ -z ${LIST+x} ]; then echo "LIST is unset"; export LIST=in_list ; else echo "LIST is set to '$LIST'"; fi



# Here we assume that our each line of our LIST contains 
# data for our program.
# Grab the line  
    export input=`head -n $SUB_ID $SLURM_SUBMIT_DIR/$LIST | tail -1`
    printenv > envs

# Run our job
	$SLURM_SUBMIT_DIR/tymer timer start_time
	echo $input > input
	$SLURM_SUBMIT_DIR/$EXE `echo $input` > output
	hostname > node
	$SLURM_SUBMIT_DIR/tymer timer end_time


:<<++++

Example output

el2:collect> ./doarray.py hpcapps
COMMAND:
sbatch -A hpcapps --array=1-24 uselist.sh
Submitted batch job 5401146

el2:collect> squeue -u tkaiser2
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
    5401146_[1-24]     short array_jo tkaiser2 PD       0:00      1 (Priority) 
el2:collect> 
el2:collect> 
el2:collect> cd 5401146
el2:5401146> ls
1  10  11  12  13  14  15  16  17  18  19  2  20  21  22  23  24  3  4  5  6  7  8  9
el2:5401146> cd 10
el2:10> ls
envs  input  myscript  node  output  timer
el2:10> cat input
7 86 11 91 400
el2:10> cat output
7 86 11 91 400 
section 1 start time= 6.6042e-05   end time=    0.50803  error= 5.58091e-09
section 2 start time= 6.7949e-05   end time=    0.50685  error= 1.4797e-10
section 3 start time=    0.50725   end time=     1.0149  error= 1.91875e-09
section 4 start time=    0.50843   end time=     1.0167  error= 1.81054e-10
el2:10> 
el2:10> 
el2:10> cd ../22
el2:22> ls
envs  input  myscript  node  output  timer
el2:22> cat input
8 24 82 13 400
el2:22> cat output
8 24 82 13 400 
section 1 start time= 6.5088e-05   end time=     0.5082  error= 3.68994e-09
section 2 start time= 6.6996e-05   end time=    0.50712  error= 8.66915e-11
section 3 start time=    0.50752   end time=     1.0152  error= 1.3155e-10
section 4 start time=     0.5086   end time=     1.0164  error= 2.47734e-09
el2:22> 
 
++++

