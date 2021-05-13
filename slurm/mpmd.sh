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

Here we look at launching Multi Program Multi Data run.   We
do this in two ways.  We use a the --multi-prog option with 
srun.  This involves creating a config_file that lists the 
programs we are going to run along with the task ID.  See:

https://computing.llnl.gov/tutorials/linux_clusters/multi-prog.html

for a quick description of the format for the config_file.

Here we create the file on the fly but it could be done
beforehand.

We have two MPI programs to run together c_ex02 and f_ex02.
They are actually the same program written in C and Fortran.
But normally MPMD applications would maybe run a GUI or a 
manager for  one task and rest workers.  The syntax here is

srun --multi-prog mapfile

where mapfile is the config_file.

It is possible to pass different arguments to each program
as discussed in the link above.

We also launch the application with mpiexec.  In this case
we just list the applications on the command line with the
number of each as would normally be done with a ":" between
listings.

These two methods both work with IntelMPI and MPT.

It is possible to do a maplist with mpiexec but that is beyound
the scope here.  Contact the author for more information.

++++


# load our version of MPI
module load mpt

#create our config_file
echo "#" $SLURM_JOBID " mapfile "> mapfile
app1=./c_ex02
for n in 0 2 4 6 ; do
  echo $n $app1 >> mapfile
done
app2=./f_ex02
for n in 1 3 5 7 ; do
  echo $n $app2 >> mapfile
done

#copy it to stdout
cat mapfile

#Run with it
srun -n8 --multi-prog mapfile > used_mapfile

# Run using mpiexec 4 copies of each app
# The next line prevents warnings when using mpiexec with IntelMPI
unset I_MPI_PMI_LIBRARY
# THIS DOES NOT WORK WITH MPT SO WE DON'T RUN IT.
# THE MAN PAGE SAYS IT SHOULD. 	INSTEAD IT RUNS
# 8 COPIES OF c_ex02.
# WE SKIP IT BY CHECKING IF MODULE mpt IS LOADED.
ml  2>&1 | grep mpt > /dev/null
if [ $? -eq 1 ] ; then
  # mpt not loaded, safe to run
  mpiexec -n 4 ./c_ex02  : -n 4 ./f_ex02 > used_mpiexec 
else
  echo "+++++++++++++++++++++++++++++++" > used_mpiexec
  echo "mpiexec -n 4 ./c_ex02  : -n 4 ./f_ex02 NOT RUN WITH MPT" >> used_mpiexec
  echo "See the script." >> used_mpiexec
  echo "+++++++++++++++++++++++++++++++" >> used_mpiexec
fi




# sort and reprint our output
echo
echo used_mapfile
grep Hello used_mapfile | sort

echo
echo used_mpiexec
grep Hello used_mpiexec | sort


:<<++++

el2:collect> sbatch -A hpcapps --partition=debug mpmd.sh 
Submitted batch job 5370453
el2:collect> el2:collect> ls -lt | head
total 2516
-rw-rw----. 1 tkaiser2 tkaiser2    1000 Dec 18 13:11 5370453.out
-rw-rw----. 1 tkaiser2 tkaiser2     450 Dec 18 13:11 used_mpiexec
-rw-rw----. 1 tkaiser2 tkaiser2     470 Dec 18 13:11 used_mapfile
-rw-rw----. 1 tkaiser2 tkaiser2     108 Dec 18 13:11 mapfile
-rw-rw----. 1 tkaiser2 tkaiser2       0 Dec 18 13:11 5370453.err
-rwxrwx---. 1 tkaiser2 tkaiser2   32600 Dec 18 13:10 c_ex02
-rwxrwx---. 1 tkaiser2 tkaiser2  853656 Dec 18 13:10 f_ex02
el2:collect> cat 5370453.out
# 5370453  mapfile 
0 ./c_ex02
2 ./c_ex02
4 ./c_ex02
6 ./c_ex02
1 ./f_ex02
3 ./f_ex02
5 ./f_ex02
7 ./f_ex02

used_mapfile
 Hello from c process: 0  Numprocs is 8
 Hello from c process: 2  Numprocs is 8
 Hello from c process: 4  Numprocs is 8
 Hello from c process: 6  Numprocs is 8
 Hello from fortran process:            1  Numprocs is            8
 Hello from fortran process:            3  Numprocs is            8
 Hello from fortran process:            5  Numprocs is            8
 Hello from fortran process:            7  Numprocs is            8

used_mpiexec
 Hello from c process: 0  Numprocs is 8
 Hello from c process: 1  Numprocs is 8
 Hello from c process: 2  Numprocs is 8
 Hello from c process: 3  Numprocs is 8
 Hello from fortran process:            4  Numprocs is            8
 Hello from fortran process:            5  Numprocs is            8
 Hello from fortran process:            6  Numprocs is            8
 Hello from fortran process:            7  Numprocs is            8
el2:collect> 


++++
