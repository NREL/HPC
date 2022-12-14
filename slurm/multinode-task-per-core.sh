#!/usr/bin/env bash
#SBATCH --nodes=2   # Change this number to get different outputs
#SBATCH --time=00:10:00
#SBATCH --job-name=node_rollcall
#SBATCH -o node_rollcall.%j # output to node_rollcall.<job ID>


:<<++++

Author: Michael Bartlett

Example of mapping an echo command to each core on an arbitrary amount
of nodes using slurm. Each node on Eagle has 36 cores, so there should
be an entry for 36 * N CPU ranks in the job output.

USAGE: sbatch –A <project_handle> -N <node amount> multinode-task-per-core.sh
++++

PROCS=$(($SLURM_NNODES * $SLURM_CPUS_ON_NODE)) # Number of CPUs * number of nodes

# Master node in jobs with N > 1 runs these
echo "I am node $SLURMD_NODENAME and I am the master node of this job with ID $SLURM_NODEID"
echo "There are $SLURM_NNODES nodes in this job, and each has $SLURM_CPUS_ON_NODE cores, for a total of $PROCS cores."
printf "Let's get each node in the job to introduce itself:\n\n"

# Send an in-line bash script to each node to run. The single quotes prevent $var evaluation.
# `srun` uses all resources by default
srun bash <<< 'printf "\tI am $SLURMD_NODENAME, my ID for this job is $SLURM_NODEID\n"' &
wait

# Do the same, but get each node to get each core to print its "rank" (unique index)
printf "\nLet's get each node to print the ranks of all their cores (concurrently!):\n\n"
srun --ntasks=$PROCS \
     bash <<< 'printf "n${SLURM_NODEID}:c"; awk "{print \$39}" /proc/self/stat' | tr '\n' ' '
echo


###################################################################
:<<++++
Example Output

I am node r5i0n13 and I am the master node of this job with ID 0
There are 2 nodes in this job, and each has 36 cores, for a total of 72 cores.
Let's get each node in the job to introduce itself:

	I am r5i0n14, my ID for this job is 1
	I am r5i0n13, my ID for this job is 0

Let's get each node to print the ranks of all their cores (concurrently!):

n1:c32 n0:c19 n0:c22 n1:c0 n0:c23 n1:c3 n0:c6 n1:c24 n0:c24 n1:c27 n0:c25 n1:c26 n0:c26 n1:c4 n0:c2 n1:c8 n0:c8 n1:c9 n0:c0 n1:c2 n0:c4 n1:c5 n0:c21 n1:c19 n0:c20 n1:c23 n0:c18 n1:c6 n0:c10 n1:c7 n0:c29 n0:c28 n0:c14 n0:c13 n0:c30 n0:c27 n1:c21 n0:c12 n1:c28 n0:c7 n0:c1 n0:c5 n0:c31 n1:c11 n1:c13 n1:c33 n1:c1 n1:c25 n0:c3 n1:c30 n1:c31 n1:c14 n1:c20 n1:c29 n1:c12 n1:c18 n1:c22 n1:c10 n1:c15 n0:c9 n0:c32 n0:c11 n0:c17 n0:c16 n0:c15 n1:c17 n0:c34 n1:c35 n1:c34 n1:c16 n0:c33 n0:c35
++++
