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

Script that runs a hybrid MPI/OpenMP program "phostone" on the
specified number of --nodes=1 with a given number of OpenMP
threads.   

Here we run with 4 tasks, 2 per node and various numbers of 
threads by setting OMP_NUM_THREADS.

If we run
the program phostone with the -F option we will get a report of
mappings of MPI tasks and threads to cores.  

task    thread             node name  first task    # on node  core
0000      0011               r4i7n35        0000         0000  0011
0000      0015               r4i7n35        0000         0000  0015
...
...
0001      0017               r4i7n35        0000         0001  0035

We set 

OMP_PLACES=cores 
OMP_PROC_BIND=spread

to ensure good mappings of threads to cores.  See the script affinity.sh
for more information on these settings.



USAGE:
    sbatch -A hpcapps --partition=debug affinity.sh 


++++

# Function for printing and executing commands
cmd() {
  echo "+ $@";
  eval "$@";
}

# load our version of MPI
module load comp-intel
module load mpt
module load gcc

# Go to the directory from which our job was launched
cd $SLURM_SUBMIT_DIR

echo running glorified hello world

# give a good mapping of threads to cores
export OMP_PLACES=cores 
export OMP_PROC_BIND=spread


for nt in 1 4 6 9 12 18 ; do 
    export OMP_NUM_THREADS=$nt
    echo OMP_NUM_THREADS=$OMP_NUM_THREADS $SLURM_JOBID.$OMP_NUM_THREADS
    CORE=36
    TPN=`echo $((CORE / nt))`
    srun --nodes=2  --tasks-per-node=$TPN  --cpus-per-task=$OMP_NUM_THREADS ./phostone -t 10 -F >> $SLURM_JOBID.$OMP_NUM_THREADS
# sort the output, first grab the header
	grep core $SLURM_JOBID.$OMP_NUM_THREADS
	grep "      r" $SLURM_JOBID.$OMP_NUM_THREADS | sort -k3,3 -k6,6
        cmd "cat $SLURM_JOBID.$OMP_NUM_THREADS | awk '{print \$3, \$6}' | grep ^r | sort -u | wc " >> $SLURM_JOBID.report
done

cat $SLURM_JOBID.report

:<<++++
Example Output:
el2:collect> sbatch -A hpcapps --partition=debug hybrid.sh 
Submitted batch job 633336

el2:slurm> ls -lt *633336*
-rw-rw----. 1 tkaiser2 tkaiser2  5029 Mar 23 13:18 6333364.18
-rw-rw----. 1 tkaiser2 tkaiser2 30522 Mar 23 13:18 6333364.out
-rw-rw----. 1 tkaiser2 tkaiser2   536 Mar 23 13:18 6333364.report
-rw-rw----. 1 tkaiser2 tkaiser2  5029 Mar 23 13:18 6333364.12
-rw-rw----. 1 tkaiser2 tkaiser2  5029 Mar 23 13:17 6333364.9
-rw-rw----. 1 tkaiser2 tkaiser2  5029 Mar 23 13:17 6333364.6
-rw-rw----. 1 tkaiser2 tkaiser2  5029 Mar 23 13:17 6333364.4
-rw-rw----. 1 tkaiser2 tkaiser2  5029 Mar 23 13:17 6333364.1
-rw-rw----. 1 tkaiser2 tkaiser2    87 Mar 23 13:17 6333364.err
el2:slurm> 
el2:slurm> cat 6333364.1 | sort -k3,3 -k6,6 | tail -5
0029      0000               r1i7n35        0000         0029  0032
0031      0000               r1i7n35        0000         0031  0033
0033      0000               r1i7n35        0000         0033  0034
0035      0000               r1i7n35        0000         0035  0035
MPI VERSION UNDEFINED - consider upgrading
el2:slurm> 

el2:slurm> cat 6333364.6 | sort -k3,3 -k6,6 | head -6
total time     10.006
task    thread             node name  first task    # on node  core
0006      0000               r104u33        0006         0000  0000
0006      0001               r104u33        0006         0000  0001
0006      0002               r104u33        0006         0000  0002
0006      0003               r104u33        0006         0000  0003
el2:slurm> 

el2:slurm> el2:slurm> cat 6333364.6 | sort -k3,3 -k6,6 | tail -5
0005      0002               r1i7n35        0000         0005  0032
0005      0003               r1i7n35        0000         0005  0033
0005      0004               r1i7n35        0000         0005  0034
0005      0005               r1i7n35        0000         0005  0035
MPI VERSION UNDEFINED - consider upgrading
el2:slurm> 

el2:slurm> cat 6333364.18 | sort -k3,3 -k6,6 | tail -5
0001      0014               r1i7n35        0000         0001  0032
0001      0015               r1i7n35        0000         0001  0033
0001      0016               r1i7n35        0000         0001  0034
0001      0017               r1i7n35        0000         0001  0035
MPI VERSION UNDEFINED - consider upgrading
el2:slurm> 

el2:slurm> cat 6333364.report
+ cat 6333364.1 | awk '{print $3, $6}' | grep ^r | sort -u | wc 
     72     144     936
+ cat 6333364.4 | awk '{print $3, $6}' | grep ^r | sort -u | wc 
     72     144     936
+ cat 6333364.6 | awk '{print $3, $6}' | grep ^r | sort -u | wc 
     72     144     936
+ cat 6333364.9 | awk '{print $3, $6}' | grep ^r | sort -u | wc 
     72     144     936
+ cat 6333364.12 | awk '{print $3, $6}' | grep ^r | sort -u | wc 
     72     144     936
+ cat 6333364.18 | awk '{print $3, $6}' | grep ^r | sort -u | wc 
     72     144     936
el2:slurm> 

++++
