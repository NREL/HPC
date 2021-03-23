#!/bin/bash 
#SBATCH --job-name="sample"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --out=%J.out
#SBATCH --error=%J.err
#SBATCH --cpus-per-task=18



:<<++++

Author: Tim Kaiser

Script that runs a hybrid MPI/OpenMP program "phostone" on the
specified number of --nodes=1 with a given number of OpenMP
threads.   

Here we run with 2 tasks and 18 threads.  We "default" to 18
threads because we have --cpus-per-task=18.  This can also be
set with the environmental variable OMP_NUM_THREADS.

If you set OMP_NUM_THREADS within a script you should add the
--cpus-per-task=$OMP_NUM_THREADS to your srun command line.

In this version we show the effects of setting the variable
KMP_AFFINITY. KMP_AFFINITY is used to control mappings of 
threads to cores when the Intel compilers are used.   

The issue is that we can, if not set, see multiple threads or
tasks end up on the same core.  We will look at three settings

If  

KMP_AFFINITY=verbose

a report will be sent to stderr, %J.err in our case where %J 
is the job number.  The mapping of threads to cores is "default"
which is somewhat arbitrary.


KMP_AFFINITY=verbose,scatter
  and 
KMP_AFFINITY=verbose,compact

We still get the report but the system tries to not map multiple
threads to the same core.  

For each run we put the output in a seperate file.  If we run
the program phostone with the -F option we will get a report of
mappings of MPI tasks and threads to cores.  

task    thread             node name  first task    # on node  core
0000      0011               r4i7n35        0000         0000  0011
0000      0015               r4i7n35        0000         0000  0015
...
...
0001      0017               r4i7n35        0000         0001  0035

We have the MPI task and OMP_THREAD_NUMBER, the node and the core number.  
Ideally, for each node each core would only have a single thread.  

While we can look at the individual output files to see if there is
duplication we actually we use the code in the nested for loops to
find and report where we have cores over/under loaded.  We note that
this only happend when we have KMP_AFFINITY=verbose


The variable KMP_AFFINITY is unique to Intel compilers.  There are
similar "OMP" variables that work for GGC compilers and Intel compilers.
For example the following settings give similar results to KMP_AFFINITY=scatter

OMP_PLACES=cores 
OMP_PROC_BIND=spread


USAGE:
    sbatch -A hpcapps --partition=debug affinity.sh 


++++


# load our version of MPI
module load mpt

# Go to the directory from which our job was launched
cd $SLURM_SUBMIT_DIR

echo running glorified hello world

1>&2 echo  "***** running verbose *****"
export KMP_AFFINITY=verbose
srun --ntasks=2   ./phostone -t 10 -F > $SLURM_JOBID.noaffinity

1>&2 echo  "***** running verbose,scatter *****"
export KMP_AFFINITY=verbose,scatter
srun --ntasks=2   ./phostone -t 10 -F > $SLURM_JOBID.scatter

1>&2 echo  "***** running verbose,compact *****"
export KMP_AFFINITY=verbose,compact
srun --ntasks=2   ./phostone -t 10 -F > $SLURM_JOBID.compact


for f in noaffinity scatter compact ; do 
  echo "Core report - cores over/under loaded KMP_AFFINITY=" $f
  for c in `seq -w 0 35` ; do  
    echo -n "$c " 
    grep 00$c\$ $SLURM_JOBID.$f | wc -l
    done | grep -v 1$
  echo " "
done


:<<++++
Example Output:


el2:collect> sbatch -A hpcapps --partition=short affinity.sh 
Submitted batch job 5368560
el2:collect> 

el2:collect> ls -l 5368560*
-rw-rw----. 1 tkaiser2 tkaiser2  2600 Dec 18 07:32 5368560.compact
-rw-rw----. 1 tkaiser2 tkaiser2 22175 Dec 18 07:32 5368560.err
-rw-rw----. 1 tkaiser2 tkaiser2  2600 Dec 18 07:32 5368560.noaffinity
-rw-rw----. 1 tkaiser2 tkaiser2   136 Dec 18 07:32 5368560.out
-rw-rw----. 1 tkaiser2 tkaiser2  2600 Dec 18 07:32 5368560.scatter
el2:collect> cat 5368560.out
running glorified hello world
Core report - cores over/under loaded
noaffinity
24 2
25 3
27 2
31 0
32 0
33 0
34 0
 
scatter
 
compact
 
el2:collect> head 5368560.err
running verbose 
OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-17
OMP: Info #214: KMP_AFFINITY: decoding x2APIC ids.
OMP: Info #156: KMP_AFFINITY: 18 available OS procs
OMP: Info #157: KMP_AFFINITY: Uniform topology
OMP: Info #285: KMP_AFFINITY: topology layer "LL cache" is equivalent to "socket".
OMP: Info #285: KMP_AFFINITY: topology layer "L3 cache" is equivalent to "socket".
OMP: Info #285: KMP_AFFINITY: topology layer "L2 cache" is equivalent to "core".
OMP: Info #285: KMP_AFFINITY: topology layer "L1 cache" is equivalent to "core".
OMP: Info #285: KMP_AFFINITY: topology layer "thread" is equivalent to "core".
el2:collect> head 5368560.noaffinity
MPI VERSION Intel(R) MPI Library 2019 Update 7 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0001               r1i1n15        0000         0000  0005
0000      0015               r1i1n15        0000         0000  0004
0000      0003               r1i1n15        0000         0000  0007
0000      0007               r1i1n15        0000         0000  0009
0000      0014               r1i1n15        0000         0000  0006
0000      0016               r1i1n15        0000         0000  0003
0000      0006               r1i1n15        0000         0000  0010
el2:collect> head 5368560.scatter
MPI VERSION Intel(R) MPI Library 2019 Update 7 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0013               r1i1n15        0000         0000  0013
0000      0012               r1i1n15        0000         0000  0012
0000      0003               r1i1n15        0000         0000  0003
0000      0007               r1i1n15        0000         0000  0007
0000      0008               r1i1n15        0000         0000  0008
0000      0016               r1i1n15        0000         0000  0016
0000      0006               r1i1n15        0000         0000  0006
el2:collect> head 5368560.compact 
MPI VERSION Intel(R) MPI Library 2019 Update 7 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0003               r1i1n15        0000         0000  0003
0000      0012               r1i1n15        0000         0000  0012
0000      0002               r1i1n15        0000         0000  0002
0000      0014               r1i1n15        0000         0000  0014
0000      0001               r1i1n15        0000         0000  0001
0000      0004               r1i1n15        0000         0000  0004
0000      0015               r1i1n15        0000         0000  0015
el2:collect> 


++++
