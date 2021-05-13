#!/bin/bash
#SBATCH --job-name="two"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00

:<<++++

Author: Tim Kaiser

We want to launch multiple instance of a program.  First we
launch two simultaneous instances.  Then we launch three
in sequence.

If we run the program phostone with the -F option we will get a report of
mappings of MPI tasks and threads to cores.  

task    thread             node name  first task    # on node  core
0000      0011               r4i7n35        0000         0000  0011
0000      0015               r4i7n35        0000         0000  0015
...
...
0001      0017               r4i7n35        0000         0001  0035

With the -t option it will run for the given number of seconds.

tymer is a glorified wallclock timer.  Its first argument is a file
in which to save data the rest of the arguments are comments put in
the file along with the time.

We first launch two instance of phostone and put them in the background.  

We NEED the wait command.  The waits for the two jobs to finish.
Without it the job may exit and the jobs could continue to run.

There is a chance that the two instance of phostone will get the same 
cores.  See multimax.sh for a way to force tasks to specific cores.

Finally we run the application three times in a loop with different
inputs. 

++++

# tymer nedas a recent version of python
module load conda

# needed for programs that use Intel threading
module load comp-intel
# load our version of MPI
module load mpt

export OMP_NUM_THREADS=4
#export OMP_PLACES=cores 
#export OMP_PROC_BIND=spread


# Phostone.c is hello world on steroids and can be 
# found at source/phostone.c.  
# tymer is a glorified wallclock timer.  


rm -rf mytimes
# The file mytimes will have a record of what happened

# Start the first job and put it the background 
./tymer mytimes start run1
srun -n 2 ./phostone -F -t 30 -T > $SLURM_JOBID.run1 &

# Start the second job and put it the background 
./tymer mytimes start run2
srun -n 2 ./phostone -F -t 10 -T > $SLURM_JOBID.run2 &

# We call the wait command to wait for the jobs
# started above to finish
./tymer mytimes start waiting
wait

# Done - record our final time
./tymer mytimes done set 1

# Now we run the application three times in a loop 
# with different inputs. 

export OMP_PLACES=cores 
export OMP_PROC_BIND=spread

for t in 5 10 15 ; do
    ./tymer mytimes doing $t
    srun -n 2 ./phostone -F -t $t -T > $SLURM_JOBID.$t
    ./tymer mytimes done $t
done


:<<++++

Example output

el2:collect> !sbatch
sbatch -A hpcapps --partition=debug multi.sh
Submitted batch job 5380554
el2:collect> cat mytimes
1608330910.230274 Fri Dec 18 15:35:10 2020      0.000      0.000 start run1
1608330910.322142 Fri Dec 18 15:35:10 2020      0.092      0.092 start run2
1608330910.386550 Fri Dec 18 15:35:10 2020      0.064      0.156 start waiting
1608330941.063865 Fri Dec 18 15:35:41 2020     30.677     30.834 done set 1
1608330941.131300 Fri Dec 18 15:35:41 2020      0.067     30.901 doing 5
1608330947.270414 Fri Dec 18 15:35:47 2020      6.139     37.040 done 5
1608330947.333237 Fri Dec 18 15:35:47 2020      0.063     37.103 doing 10
1608330959.771684 Fri Dec 18 15:35:59 2020     12.438     49.541 done 10
1608330959.834768 Fri Dec 18 15:35:59 2020      0.063     49.604 doing 15
1608330977.523985 Fri Dec 18 15:36:17 2020     17.689     67.294 done 15
el2:collect> ls *5380554*
5380554.10  5380554.15  5380554.5  5380554.run1  5380554.run2  slurm-5380554.out
el2:collect> ls -l *5380554*
-rw-rw----. 1 tkaiser2 tkaiser2 746 Dec 18 15:35 5380554.10
-rw-rw----. 1 tkaiser2 tkaiser2 746 Dec 18 15:36 5380554.15
-rw-rw----. 1 tkaiser2 tkaiser2 746 Dec 18 15:35 5380554.5
-rw-rw----. 1 tkaiser2 tkaiser2 746 Dec 18 15:35 5380554.run1
-rw-rw----. 1 tkaiser2 tkaiser2 746 Dec 18 15:35 5380554.run2
-rw-rw----. 1 tkaiser2 tkaiser2 746 Dec 18 15:36 slurm-5380554.out
el2:collect> cat 5380554.run1
Fri Dec 18 15:35:10 2020
MPI VERSION Intel(R) MPI Library 2019 Update 7 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0000               r3i7n35        0000         0000  0017
0000      0003               r3i7n35        0000         0000  0000
0000      0001               r3i7n35        0000         0000  0001
0000      0002               r3i7n35        0000         0000  0002
0001      0000               r3i7n35        0000         0001  0035
0001      0002               r3i7n35        0000         0001  0020
0001      0003               r3i7n35        0000         0001  0019
0001      0001               r3i7n35        0000         0001  0021
total time     30.002
Fri Dec 18 15:35:40 2020
el2:collect> cat 5380554.run2
Fri Dec 18 15:35:10 2020
MPI VERSION Intel(R) MPI Library 2019 Update 7 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0002               r3i7n35        0000         0000  0006
0000      0003               r3i7n35        0000         0000  0004
0000      0000               r3i7n35        0000         0000  0003
0000      0001               r3i7n35        0000         0000  0005
0001      0003               r3i7n35        0000         0001  0022
0001      0000               r3i7n35        0000         0001  0018
0001      0002               r3i7n35        0000         0001  0024
0001      0001               r3i7n35        0000         0001  0023
total time     10.009
Fri Dec 18 15:35:20 2020
el2:collect> 

++++




