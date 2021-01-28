#!/bin/bash
#SBATCH --job-name="hostname"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=8
## ask for 10 minutes
#SBATCH --time=00:10:00

:<<++++

Author: Tim Kaiser

Script that runs a serial command "hostname" on the
specified number of nodes and with the specified number of
instances.  The output will go into the slurm default output
file slurm-xxxxxx.out where xxxxxx is the job id.

Here we ask for 2 nodes and 4 tasks per node.  Specifing 
ntasks=8 is redundent but if you specifiy all three parameters
they must be consistent.

Usage:
sbatch –A=myaccount --partition=debug  hostname.sh

++++

# Go to the directoy from which our job was launched
cd $SLURM_SUBMIT_DIR

#run an application
srun hostname
date

:<<++++
Example Output:

cat slurm-5357831.out
r105u33
r105u37
r105u33
r105u33
r105u33
r105u37
r105u37
r105u37
Thu Jan  7 10:46:36 MST 2021
++++
