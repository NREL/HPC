#!/bin/bash
#SBATCH --job-name="mpi4py"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:01:00
#SBATCH --partition=debug
#SBATCH --account=hpcapps

# Launching multiple mpi4py programs on a collection of nodes.


:<<++++

Author: Ethan Young, Tim Kaiser

This script is show two ways to map MPI tasks to node when we
have multiple mpi applications running simultaneously on a
set of nodes.  

The first method uses a combination of the two options: distribution 
and nodelist.

From the srun man page we have:

distribution
     Specify alternate distribution methods for remote processes. 
     
nodelist
      Request a specific list of hosts.
      
The second method uses the relative option.  Again from the srun
man page we have:

relative
      Run a job step relative to node n of the current allocation.

We assume we are running on two or more nodes.

The application we are running is a python mpi4py code, report.py.  
We assume we are using mpt MPI for the backend.  The script jupyter.sh
can be used to create a conda environment with mpi4py with an
Intel backend in the file .  

In this script our conda environement is called "dompt" as seen in 
the line:

source activate dompt


Also, report.py calls two functions findcore and forcecore that are 
defined in the file spam.c.  After you have your conda environment
set up you can build the spam module with the command:

cd source
python setup.py install


The routines findcore and forcecore find the core on which the
calling process is running and force it to a particular core.  

This is useful in this case when launcing multiple MPI runs across
node they might get mapped to the same cores.  

report.py reads the environmental variable OFFSET to use as the 
starting core for its layout of tasks to cores.  So we set OFFSET
before lanuching our individual copies of the program.

The first set of runs is launched in nested for loops.  We launch 
a total of 4 instances with each mapped to cores[0-17] or cores[18-35]
and one of the two nodes.

In our second set of lanuches we lanuch first on the zeroth node
and then on node "one" using the relative option.  

++++



# Load my version of mpi4py.
module load conda
source activate
source activate dompt
module load gcc/8.4.0 
module load mpt


date +"%y%m%d%H%M%S"
# Get a list of our nodes.
nodes=`scontrol show hostnames`


# Run two sets of calculations on each one of our nodes.
# The inner loop below maps to a particular node.
# The sets are passed a command line argument 10 or 11
# from the outside loop.
# The first set runs on the first 18 cores and the second
# set runs on the last 18 cores.  

# The program report.py contains code to map tasks to 
# cores. It maps to core  mpi_id+OFFSET.
# For full functinality is requires installing a external
# python module.  See the source on how to do that.

# Note we are putting these MPI programs in the background
# so they run concurrently. The wait command is required.

export OFFSET=0
for i in 10 11 ; do 
    for n in $nodes ; do 
        srun   --nodes=1 --distribution=block  -n 18  --nodelist=$n ./report.py $i > $[i]_$n & 
    done
    export OFFSET=18
done
wait
date +"%y%m%d%H%M%S"


# Same general idea as above excpet we use the "relative" option instead of 
# "distribution" and we launch in groups of 4.

date +"%y%m%d%H%M%S"
export OFFSET=0
srun   --nodes=1 --relative=0  -n 4  ./report.py 10  > run0 &
export OFFSET=4
srun   --nodes=1 --relative=0  -n 4  ./report.py 10  > run1 &
export OFFSET=0
srun   --nodes=1 --relative=1  -n 4  ./report.py 10  > run2 &
export OFFSET=4
srun   --nodes=1 --relative=1  -n 4  ./report.py 10  > run3 &
wait
date +"%y%m%d%H%M%S"

# Sort our output based on the core on which a process is running
for i in 10 11 ; do
    for n in $nodes ; do
        echo $[i]_$n
        grep Hello $[i]_$n | sort -n -k8,8
    done
done

for n in run0 run1 run2 run3 ; do
    echo $n
    grep Hello $n | sort -n -k8,8
done


:<<++++

Example output:

(/home/tkaiser2/.conda-envs/newmpi) el1:mpi4py> cat slurm-5306445.out
201208073517
201208073531
201208073531
201208073542
10_r103u21
xxxxxx Hello from 0 on r103u21 , 0 10
xxxxxx Hello from 1 on r103u21 , 1 10
xxxxxx Hello from 2 on r103u21 , 2 10
xxxxxx Hello from 3 on r103u21 , 3 10
xxxxxx Hello from 4 on r103u21 , 4 10
xxxxxx Hello from 5 on r103u21 , 5 10
xxxxxx Hello from 6 on r103u21 , 6 10
xxxxxx Hello from 7 on r103u21 , 7 10
xxxxxx Hello from 8 on r103u21 , 8 10
xxxxxx Hello from 9 on r103u21 , 9 10
xxxxxx Hello from 10 on r103u21 , 10 10
xxxxxx Hello from 11 on r103u21 , 11 10
xxxxxx Hello from 12 on r103u21 , 12 10
xxxxxx Hello from 13 on r103u21 , 13 10
xxxxxx Hello from 14 on r103u21 , 14 10
xxxxxx Hello from 15 on r103u21 , 15 10
xxxxxx Hello from 16 on r103u21 , 16 10
xxxxxx Hello from 17 on r103u21 , 17 10
10_r103u23
xxxxxx Hello from 0 on r103u23 , 0 10
xxxxxx Hello from 1 on r103u23 , 1 10
xxxxxx Hello from 2 on r103u23 , 2 10
xxxxxx Hello from 3 on r103u23 , 3 10
xxxxxx Hello from 4 on r103u23 , 4 10
xxxxxx Hello from 5 on r103u23 , 5 10
xxxxxx Hello from 6 on r103u23 , 6 10
xxxxxx Hello from 7 on r103u23 , 7 10
xxxxxx Hello from 8 on r103u23 , 8 10
xxxxxx Hello from 9 on r103u23 , 9 10
xxxxxx Hello from 10 on r103u23 , 10 10
xxxxxx Hello from 11 on r103u23 , 11 10
xxxxxx Hello from 12 on r103u23 , 12 10
xxxxxx Hello from 13 on r103u23 , 13 10
xxxxxx Hello from 14 on r103u23 , 14 10
xxxxxx Hello from 15 on r103u23 , 15 10
xxxxxx Hello from 16 on r103u23 , 16 10
xxxxxx Hello from 17 on r103u23 , 17 10
11_r103u21
xxxxxx Hello from 0 on r103u21 , 18 11
xxxxxx Hello from 1 on r103u21 , 19 11
xxxxxx Hello from 2 on r103u21 , 20 11
xxxxxx Hello from 3 on r103u21 , 21 11
xxxxxx Hello from 4 on r103u21 , 22 11
xxxxxx Hello from 5 on r103u21 , 23 11
xxxxxx Hello from 6 on r103u21 , 24 11
xxxxxx Hello from 7 on r103u21 , 25 11
xxxxxx Hello from 8 on r103u21 , 26 11
xxxxxx Hello from 9 on r103u21 , 27 11
xxxxxx Hello from 10 on r103u21 , 28 11
xxxxxx Hello from 11 on r103u21 , 29 11
xxxxxx Hello from 12 on r103u21 , 30 11
xxxxxx Hello from 13 on r103u21 , 31 11
xxxxxx Hello from 14 on r103u21 , 32 11
xxxxxx Hello from 15 on r103u21 , 33 11
xxxxxx Hello from 16 on r103u21 , 34 11
xxxxxx Hello from 17 on r103u21 , 35 11
11_r103u23
xxxxxx Hello from 0 on r103u23 , 18 11
xxxxxx Hello from 1 on r103u23 , 19 11
xxxxxx Hello from 2 on r103u23 , 20 11
xxxxxx Hello from 3 on r103u23 , 21 11
xxxxxx Hello from 4 on r103u23 , 22 11
xxxxxx Hello from 5 on r103u23 , 23 11
xxxxxx Hello from 6 on r103u23 , 24 11
xxxxxx Hello from 7 on r103u23 , 25 11
xxxxxx Hello from 8 on r103u23 , 26 11
xxxxxx Hello from 9 on r103u23 , 27 11
xxxxxx Hello from 10 on r103u23 , 28 11
xxxxxx Hello from 11 on r103u23 , 29 11
xxxxxx Hello from 12 on r103u23 , 30 11
xxxxxx Hello from 13 on r103u23 , 31 11
xxxxxx Hello from 14 on r103u23 , 32 11
xxxxxx Hello from 15 on r103u23 , 33 11
xxxxxx Hello from 16 on r103u23 , 34 11
xxxxxx Hello from 17 on r103u23 , 35 11
run0
xxxxxx Hello from 0 on r103u21 , 0 10
xxxxxx Hello from 1 on r103u21 , 1 10
xxxxxx Hello from 2 on r103u21 , 2 10
xxxxxx Hello from 3 on r103u21 , 3 10
run1
xxxxxx Hello from 0 on r103u21 , 4 10
xxxxxx Hello from 1 on r103u21 , 5 10
xxxxxx Hello from 2 on r103u21 , 6 10
xxxxxx Hello from 3 on r103u21 , 7 10
run2
xxxxxx Hello from 0 on r103u23 , 0 10
xxxxxx Hello from 1 on r103u23 , 1 10
xxxxxx Hello from 2 on r103u23 , 2 10
xxxxxx Hello from 3 on r103u23 , 3 10
run3
xxxxxx Hello from 0 on r103u23 , 4 10
xxxxxx Hello from 1 on r103u23 , 5 10
xxxxxx Hello from 2 on r103u23 , 6 10
xxxxxx Hello from 3 on r103u23 , 7 10
(/home/tkaiser2/.conda-envs/newmpi) el1:mpi4py> 

++++
