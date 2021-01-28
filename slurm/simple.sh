#!/bin/bash 
#SBATCH --job-name="sample"
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --out=%J.out
#SBATCH --error=%J.err


:<<++++

Author: Tim Kaiser

Script that runs a MPI program "purempi" on the
specified number of --nodes=2.  This can be changed
in the script or specifying a different number on the
command line.  

We first run specifying the number of tasks on the srun
line.  Note that we have not specified an output file
on the command line.  In the script header we have the
line: #SBATCH --out=%J.out  This indicates that the
output will go to a file %J.out where %J is replaced
with the job number.  Any error report will go to 
%J.err as specified in the header.  

In our second run we do not specify the numger of tasks.
It defaults to 1.

We do however, pipe data into a file $SLURM_JOBID.stdout
where SLURM_JOBID is again the job number.  

purempi run with the options -t 10 -T will run for 10 seconds
and print a list of nodes on which it is run and its start and
stop time.

Note the date command output will go into %J.out and %J.err should
be empty.


USAGE:
    sbatch -A hpcapps --partition=debug simple.sh 


++++


# load our version of MPI
module load mpt

# Go to the directory from which our job was launched
cd $SLURM_SUBMIT_DIR

echo running glorified hello world

date +"starting %y-%m-%d %H:%M:%S"
echo "first run"
srun --ntasks=4 --cpus-per-task=2  ./purempi -t 10 -T -F

date +"starting %y-%m-%d %H:%M:%S"
echo "second run"
srun   ./purempi -t 10 -T -F > $SLURM_JOBID.stdout
date +"finished %y-%m-%d %H:%M:%S"

:<<++++
Example Output:

el1:collect> sbatch -A hpcapps --partition=debug simple.sh 
Submitted batch job 5361993
el1:collect> 

el1:collect> ls -l 5361993*
-rw-rw----. 1 tkaiser2 tkaiser2   0 Dec 17 10:21 5361993.err
-rw-rw----. 1 tkaiser2 tkaiser2 236 Dec 17 10:21 5361993.out
-rw-rw----. 1 tkaiser2 tkaiser2  88 Dec 17 10:21 5361993.stdout
el1:collect> cat 5361993.out
running glorified hello world
starting 20-12-17 10:21:04
first run
Thu Dec 17 10:21:05 2020
r105u37
r105u33
r105u33
r105u37
total time     10.005
Thu Dec 17 10:21:15 2020
starting 20-12-17 10:21:15
second run
finished 20-12-17 10:21:27
el1:collect> 
el1:collect> 

el1:collect> cat 5361993.stdout
Thu Dec 17 10:21:17 2020
r105u37
r105u33
total time     10.007
Thu Dec 17 10:21:27 2020
el1:collect> 
++++
