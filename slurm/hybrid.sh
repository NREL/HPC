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


# load our version of MPI
module load mpt

# Go to the directory from which our job was launched
cd $SLURM_SUBMIT_DIR

echo running glorified hello world

# give a good mapping of threads to cores
export OMP_PLACES=cores 
export OMP_PROC_BIND=spread


for nt in 1 4 9 ; do 
    export OMP_NUM_THREADS=$nt
    echo OMP_NUM_THREADS=$OMP_NUM_THREADS $SLURM_JOBID.$OMP_NUM_THREADS
    srun --ntasks=4  --tasks-per-node=2  ./phostone -t 10 -F >> $SLURM_JOBID.$OMP_NUM_THREADS
# sort the output, first grab the header
	grep core $SLURM_JOBID.$OMP_NUM_THREADS
	grep "      r" $SLURM_JOBID.$OMP_NUM_THREADS | sort -k3,3 -k6,6
done


:<<++++
Example Output:
el2:collect> sbatch -A hpcapps --partition=debug hybrid.sh 
Submitted batch job 5368687

el2:collect> ls -l *5368687*
-rw-rw----. 1 tkaiser2 tkaiser2  424 Dec 18 07:57 5368687.1
-rw-rw----. 1 tkaiser2 tkaiser2 1240 Dec 18 07:57 5368687.4
-rw-rw----. 1 tkaiser2 tkaiser2 2600 Dec 18 07:57 5368687.9
-rw-rw----. 1 tkaiser2 tkaiser2    0 Dec 18 07:56 5368687.err
-rw-rw----. 1 tkaiser2 tkaiser2 4126 Dec 18 07:57 5368687.out
el2:collect> cat 5368687.out
running glorified hello world
OMP_NUM_THREADS=1 5368687.1
task    thread             node name  first task    # on node  core
0000      0000               r102u34        0000         0000  0000
0001      0000               r102u34        0000         0001  0018
0002      0000               r102u35        0002         0000  0000
0003      0000               r102u35        0002         0001  0018
OMP_NUM_THREADS=4 5368687.4
task    thread             node name  first task    # on node  core
0000      0000               r102u34        0000         0000  0000
0000      0001               r102u34        0000         0000  0004
0000      0002               r102u34        0000         0000  0009
0000      0003               r102u34        0000         0000  0014
0001      0000               r102u34        0000         0001  0018
0001      0001               r102u34        0000         0001  0022
0001      0002               r102u34        0000         0001  0027
0001      0003               r102u34        0000         0001  0032
0002      0000               r102u35        0002         0000  0000
0002      0001               r102u35        0002         0000  0004
0002      0002               r102u35        0002         0000  0009
0002      0003               r102u35        0002         0000  0014
0003      0000               r102u35        0002         0001  0018
0003      0001               r102u35        0002         0001  0022
0003      0002               r102u35        0002         0001  0027
0003      0003               r102u35        0002         0001  0032
OMP_NUM_THREADS=9 5368687.9
task    thread             node name  first task    # on node  core
0000      0000               r102u34        0000         0000  0000
0000      0001               r102u34        0000         0000  0002
0000      0002               r102u34        0000         0000  0004
0000      0003               r102u34        0000         0000  0006
0000      0004               r102u34        0000         0000  0008
0000      0005               r102u34        0000         0000  0010
0000      0006               r102u34        0000         0000  0012
0000      0007               r102u34        0000         0000  0014
0000      0008               r102u34        0000         0000  0016
0001      0000               r102u34        0000         0001  0018
0001      0001               r102u34        0000         0001  0020
0001      0002               r102u34        0000         0001  0022
0001      0003               r102u34        0000         0001  0024
0001      0004               r102u34        0000         0001  0026
0001      0005               r102u34        0000         0001  0028
0001      0006               r102u34        0000         0001  0030
0001      0007               r102u34        0000         0001  0032
0001      0008               r102u34        0000         0001  0034
0002      0000               r102u35        0002         0000  0000
0002      0001               r102u35        0002         0000  0002
0002      0002               r102u35        0002         0000  0004
0002      0003               r102u35        0002         0000  0006
0002      0004               r102u35        0002         0000  0008
0002      0005               r102u35        0002         0000  0010
0002      0006               r102u35        0002         0000  0012
0002      0007               r102u35        0002         0000  0014
0002      0008               r102u35        0002         0000  0016
0003      0000               r102u35        0002         0001  0018
0003      0001               r102u35        0002         0001  0020
0003      0002               r102u35        0002         0001  0022
0003      0003               r102u35        0002         0001  0024
0003      0004               r102u35        0002         0001  0026
0003      0005               r102u35        0002         0001  0028
0003      0006               r102u35        0002         0001  0030
0003      0007               r102u35        0002         0001  0032
0003      0008               r102u35        0002         0001  0034
el2:collect> 

++++
