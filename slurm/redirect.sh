#!/bin/bash
#SBATCH --job-name="atest"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j

:<<++++

Author: Tim Kaiser

This script is rather obscure.  It again runs the hello
world example, phostone.  

The purpose is to demonstrate low level redirection of
output. 

Slurm has the restriction that stdout and stderr from the
script will always go to the directory from which the 
script is launched.  

This script shows how to get around that restriction.  In
particular, we create a file in our home directory and 
stdout and stderr from slurm will go to that file.

The normal output files will still be created but they
will be empty.

This script is based on: 

http://compgroups.net/comp.unix.shell/bash-changing-stdout/497180


Note that we can still use the ">" to redirect output for
individual execuatables.  


++++


cd $SLURM_SUBMIT_DIR

#needed for threaded programs compiled with Intel compilers
module load comp-intel
# Load our version of mpi
module load mpt
export OMP_NUM_THREADS=2

###############
# http://compgroups.net/comp.unix.shell/bash-changing-stdout/497180
# set up our redirects of stdout and stderr
                            # 1 and 2 are file descriptors for  
                            # stdout and stderr
                            # 3 and 4 are descriptors to logfile
                            # we will use 3 for stdout 4 for stderr
exec 3>>$HOME/logfile.`date +"%y%m%d%H%M%S"` 
                            # anything that goes to 4 will go to 3
                            # which is our file we have created
exec 4>&3 
exec 5>&1 6>&2              # save "pointers" to stdin and stdout
exec 1>&3 2>&4              # redirect stdin and stdout to file
###############
# normal commands
                            # this line goes to stdout 
echo this is a test from stdout
                            # this line goes to stderr 
echo this is a test from stderr >&2
                            # error message goes to stderr 
ls file_that_does_not_exist
srun -n 8 ./phostone -F -t 10 > myout.$SLURM_JOBID
srun -n 8 ./phostone -F -t 10
###############
exec 1>&5 2>&6              # restore original stdin and stdout
3>&- 4>&-                   # close logfile descriptors
5>&- 6>&-                   # close saved stdin and stdout


:<<++++

Example output



el2:collect> !sbatch
sbatch -A hpcapps --partition=debug redirect.sh 
Submitted batch job 5398839
el2:collect> 
el2:collect> 
el2:collect> ls -lt *5398839*
-rw-rw----. 1 tkaiser2 tkaiser2 19736 Dec 21 09:07 myout.5398839
-rw-rw----. 1 tkaiser2 tkaiser2     0 Dec 21 09:07 stderr.5398839
-rw-rw----. 1 tkaiser2 tkaiser2     0 Dec 21 09:07 stdout.5398839
el2:collect> 
el2:collect> 
el2:collect> ls -lt ~/log*
-rw-rw----. 1 tkaiser2 tkaiser2 19940 Dec 21 09:07 /home/tkaiser2/logfile.201221090731
el2:collect> 
el2:collect> 
el2:collect> 
el2:collect> head /home/tkaiser2/logfile.201221090731
this is a test from stdout
this is a test from stderr
ls: cannot access file_that_does_not_exist: No such file or directory
-rw-rw---- 1 tkaiser2 tkaiser2 0 Dec 21 09:07 /home/tkaiser2/collect/a_new_file
MPI VERSION Intel(R) MPI Library 2019 Update 7 for Linux* OS

task    thread             node name  first task    # on node  core
0000      0008               r2i7n35        0000         0000  0020
0000      0004               r2i7n35        0000         0000  0027
0000      0016               r2i7n35        0000         0000  0001
el2:collect> 
++++
