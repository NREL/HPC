#!/bin/bash 
#SBATCH --job-name="sample"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --out=%J.out
#SBATCH --error=%J.err


:<<++++

Author: Tim Kaiser

This script shows how to set an environmental variable 
before running a script and use it from within the script.

This is useful for cases where you have a bunch of runs
with different input files or exeicutables and you do not
want to modify your script.  

Here we set the variable INPUT before we do the sbatch 
command.  INPUT points to our input file.  In our script
we check to see if it is defined and exit if not.  We
could also set a default value instead.

In our example output we show what happens if we first
run without setting INPUT and then with it set with two
different values.  

Note we save a copy of our input file with the job id
appended and lable our outpu with the job id also.
This helps us keep track of what we did.

++++


# load our version of MPI
module load mpt


if [ -z ${INPUT+x} ]; then 
    echo "var is unset - quitting"
    echo "To set:"
    echo "export INPUT=afile"
    exit
else 
    echo "INPUT is set to '$INPUT'"
fi

cp $INPUT $INPUT.$SLURM_JOBID
export EXE=stf_01

srun  ./$EXE  < $INPUT >> $EXE.$INPUT.output.$SLURM_JOBID


:<<++++

Author: Tim Kaiser


el2:collect> sbatch -A hpcapps --partition=debug fromenv.sh 
Submitted batch job 5370160
el2:collect> ls -l *5370160*
-rw-rw----. 1 tkaiser2 tkaiser2  0 Dec 18 10:25 5370160.err
-rw-rw----. 1 tkaiser2 tkaiser2 51 Dec 18 10:25 5370160.out
el2:collect> cat 5370160.out
var is unset - quitting
To set:
export INPUT=afile
el2:collect> 
el2:collect> 
el2:collect> export INPUT=st.in
el2:collect> 
el2:collect> sbatch -A hpcapps --partition=debug fromenv.sh 
Submitted batch job 5370166
el2:collect> 
el2:collect> 
el2:collect> export INPUT=st2.in
el2:collect> 
el2:collect> sbatch -A hpcapps --partition=debug fromenv.sh 
Submitted batch job 5370167
el2:collect> 
el2:collect> 
el2:collect> ls -l *5370166*
-rw-rw----. 1 tkaiser2 tkaiser2   0 Dec 18 10:27 5370166.err
-rw-rw----. 1 tkaiser2 tkaiser2  24 Dec 18 10:27 5370166.out
-rw-rw----. 1 tkaiser2 tkaiser2 365 Dec 18 10:28 stf_01.st.in.output.5370166
-rw-r-----. 1 tkaiser2 tkaiser2  54 Dec 18 10:27 st.in.5370166
el2:collect> 
el2:collect> 
el2:collect> cat 5370166.out
INPUT is set to 'st.in'
el2:collect> 
el2:collect> ls -l *5370167*
-rw-rw----. 1 tkaiser2 tkaiser2   0 Dec 18 10:28 5370167.err
-rw-rw----. 1 tkaiser2 tkaiser2  25 Dec 18 10:28 5370167.out
-rw-r-----. 1 tkaiser2 tkaiser2  54 Dec 18 10:28 st2.in.5370167
-rw-rw----. 1 tkaiser2 tkaiser2 365 Dec 18 10:29 stf_01.st2.in.output.5370167
el2:collect> 
el2:collect> cat 5370167.out
INPUT is set to 'st2.in'
el2:collect> 

++++