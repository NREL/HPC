#!/bin/bash 
#SBATCH --job-name="sample"
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --out=%J.out
#SBATCH --error=%J.err
#SBATCH --cpus-per-task=36



:<<++++

Author: Tim Kaiser

Script that runs a OpenMP program "slowinvert" on the
with a given number of OpenMP threads.   

We run it with OMP_NUM_THREADS set to 18 and 36

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

The variable KMP_AFFINITY is unique to Intel compilers.  There are
similar "OMP" variables that work for GGC compilers and Intel compilers.
For example the following settings give similar results to KMP_AFFINITY=scatter

OMP_PLACES=cores 
OMP_PROC_BIND=spread


USAGE:
    sbatch -A hpcapps --partition=debug openmp.sh 


++++


# load our version of MPI
module load mpt
module load mkl

# Go to the directory from which our job was launched
cd $SLURM_SUBMIT_DIR

echo running a matrix inversion example with one thread per inversion

1>&2 echo  "***** running verbose,scatter *****"
export KMP_AFFINITY=verbose,scatter
export OMP_NUM_THREADS=18
#piping in /dev/null to this program gives us default input
./slowinvert < /dev/null  > $SLURM_JOBID.$OMP_NUM_THREADS.scatter

export OMP_NUM_THREADS=36
#piping in /dev/null to this program gives us default input
./slowinvert < /dev/null  > $SLURM_JOBID.$OMP_NUM_THREADS.scatter

unset KMP_AFFINITY
OMP_PLACES=cores 
OMP_PROC_BIND=spread

export OMP_NUM_THREADS=18
#piping in /dev/null to this program gives us default input
./slowinvert < /dev/null  > $SLURM_JOBID.$OMP_NUM_THREADS.spread

export OMP_NUM_THREADS=36
#piping in /dev/null to this program gives us default input
./slowinvert < /dev/null  > $SLURM_JOBID.$OMP_NUM_THREADS.spread


:<<++++
Example Output:


el3:collect> sbatch -p debug -A hpcapps openmp.sh 
Submitted batch job 5637141
el3:collect> ls -lt *5637141*
-rw-rw----. 1 tkaiser2 tkaiser2  7192 Jan  5 14:06 5637141.36.spread
-rw-rw----. 1 tkaiser2 tkaiser2  7192 Jan  5 14:05 5637141.18.spread
-rw-rw----. 1 tkaiser2 tkaiser2  7192 Jan  5 14:05 5637141.36.scatter
-rw-rw----. 1 tkaiser2 tkaiser2 10969 Jan  5 14:05 5637141.err
-rw-rw----. 1 tkaiser2 tkaiser2  7192 Jan  5 14:04 5637141.18.scatter
-rw-rw----. 1 tkaiser2 tkaiser2    65 Jan  5 14:03 5637141.out
el3:collect> cat 5637141.out
running a matrix inversion example with one thread per inversion

el3:collect> head 5637141.err
***** running verbose,scatter *****
OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-35
OMP: Info #214: KMP_AFFINITY: decoding x2APIC ids.
OMP: Info #156: KMP_AFFINITY: 36 available OS procs
OMP: Info #157: KMP_AFFINITY: Uniform topology
OMP: Info #285: KMP_AFFINITY: topology layer "LL cache" is equivalent to "socket".
OMP: Info #285: KMP_AFFINITY: topology layer "L3 cache" is equivalent to "socket".
OMP: Info #285: KMP_AFFINITY: topology layer "L2 cache" is equivalent to "core".
OMP: Info #285: KMP_AFFINITY: topology layer "L1 cache" is equivalent to "core".
OMP: Info #285: KMP_AFFINITY: topology layer "thread" is equivalent to "core".
el3:collect> 

el3:collect> tail 5637141.18.scatter
   24    0   50665.757   50661.874       3.883
   16    0   50665.758   50661.869       3.889
   64    0   50665.786   50661.896       3.890
   72    0   50665.788   50661.897       3.891
   32    0   50665.821   50661.920       3.901
   56    0   50665.829   50661.930       3.899
   48    0   50665.832   50661.932       3.900
   40    0   50665.876   50661.984       3.892
    8    0   50665.886   50661.992       3.894
 invert time=      16.056

el3:collect> tail 5637141.36.scatter
   50    0   50701.399   50695.520       5.879
   46    0   50701.400   50695.506       5.894
   54    0   50701.412   50695.529       5.883
   22    0   50701.416   50695.516       5.900
   34    0   50701.421   50695.526       5.895
   70    0   50701.426   50695.561       5.865
   26    0   50701.426   50695.529       5.897
   58    0   50701.427   50695.533       5.894
    2    0   50701.538   50695.738       5.800
 invert time=      12.702

el3:collect> tail 5637141.18.spread 
   12    0   50742.040   50738.523       3.517
   28    0   50742.096   50738.539       3.557
   64    0   50742.726   50739.334       3.392
   20    0   50743.283   50739.577       3.706
   48    0   50743.338   50739.537       3.801
    4    0   50743.416   50739.629       3.787
   16    0   50743.659   50739.923       3.736
    8    0   50743.664   50739.961       3.703
   36    0   50743.761   50739.958       3.803
 invert time=      15.974

el3:collect> tail 5637141.36.spread 
   30    0   50774.025   50769.060       4.965
   42    0   50774.031   50769.106       4.925
    2    0   50774.054   50769.150       4.904
   16    0   50774.065   50769.166       4.899
   22    0   50774.074   50769.166       4.908
   28    0   50774.079   50769.178       4.901
   32    0   50774.089   50769.203       4.886
   50    0   50774.245   50769.261       4.984
   36    0   50774.345   50769.374       4.971
 invert time=      10.342
++++

