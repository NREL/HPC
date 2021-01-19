#!/bin/bash

:<<++++

Author: Tim Kaiser

This script shows how to use slurm dependencies to build
complex workflows.

(1) Starts with 5 jobs that need to run in sequence.

(2) Submits 4 jobs that will wait for the previous 5.  However,
these 4 jobs can run at the same time.  

(3) Finally a single job is run that depends or the previous 4.

The dependency graph is:

        job1 
        job2
        job3
        job4
        job5
   /   /    \   \
job6 job7 job8 job9
  \   \    /    /
      job10  
 
The way it works is that it grabs the JOBIDs returned by sbatch 
and uses then as dependencies.  For jobs[2-5] this is easy since
there is a single dependency.  For jobs[6-8] we collect dependencies
in the string myset1. We note here that we could have made  jobs[6-8]
only dependent on job5 since if job5 finishes we know the others have
completed.

For job10 we collect the dependency list in the string myset2.

The slurm script we are running is old_new.sh.  It can use the 
variables OLD_DIR and NEW_DIR to specify directories from which
to get data from a previous run and where to do the current run.

If OLD_DIR is defined old_new.sh will copy all files from that directory
to NEW_DIR.  There is also a variable, OLD_FILES, not used here that
can be used to only copy specific files.

For the first 5 jobs data goes into directories ser[1-5]. The next 4,
par[1-4] and the final in directory final.

Usage:

./FAN.sh account


++++



if [ -z ${1+x} ]; then
	echo USAGE:
	echo $0 account
	echo Your account needs to be set on the command line
	exit
fi
export ACC=$1

# Here is the script we will run
export SCRIPT=old_new.sh



rm -rf ser* par* final*
echo "Starts with 5 jobs that need to run in sequence."
unset OLD_DIR
unset OLD_FILES
export NEW_DIR=ser1
jid=`sbatch --partition=short -A $ACC $SCRIPT | awk '{print $NF }'`
echo $jid 
myset1=""
for job in ser2 ser3 ser4 ser5 ; do
  export OLD_DIR=$NEW_DIR
  export NEW_DIR=$job
  echo --dependency=afterok:$jid
  jid=`sbatch --partition=short -A $ACC --dependency=afterok:$jid $SCRIPT | awk '{print $NF }'`
  echo $jid
  myset1=$myset1,afterok:$jid
done
myset1=`echo $myset1 | sed "s/,//"`
echo $myset1

echo "Now 4 jobs that will wait for the previous 5,"
echo "however, these are independent of each other."
myset2=""
export OLD_DIR=$NEW_DIR
for job in par1 par2 par3 par4 ; do
  export NEW_DIR=$job
  echo --dependency=$myset1
  jid=`sbatch  --partition=short -A $ACC --dependency=$myset1 $SCRIPT | awk '{print $NF }'`
  echo $jid
  myset2=$myset2,afterok:$jid
done

myset2=`echo $myset2 | sed "s/,//"`
echo $myset2

echo "Finally a single job that waits for the previous 4"
getfiles=`echo $myset2 | sed "s/,/ /g"`
getfiles=`echo $getfiles | sed "s/afterok://g"`
echo $getfiles
unset OLD_DIR
export NEW_DIR=final
export OLD_FILES=$getfiles
jid=`sbatch  --partition=short -A $ACC --dependency=$myset2 $SCRIPT | awk '{print $NF }'`
echo $jid

echo +-+-+- REPORT OF PENDING JOB DEPENDECIES +-+-+-
# get a list of jobs and show Dependencies
squeue  -u $LOGNAME
for jid in `squeue -h -u $LOGNAME | awk '{print $1}'` ; do
echo job $jid
scontrol show jobid -dd $jid | grep Dependency
done



:<<++++

Example output


el2:collect> ./FAN hpcapps
Start with 5 jobs that need to run in sequence.
5400356
--dependency=afterok:5400356
5400357
--dependency=afterok:5400357
5400358
--dependency=afterok:5400358
5400359
--dependency=afterok:5400359
5400360
afterok:5400357,afterok:5400358,afterok:5400359,afterok:5400360
Now 4 jobs that will wait for the previous 5,
however, these are independent of each other.
--dependency=afterok:5400357,afterok:5400358,afterok:5400359,afterok:5400360
5400361
--dependency=afterok:5400357,afterok:5400358,afterok:5400359,afterok:5400360
5400362
--dependency=afterok:5400357,afterok:5400358,afterok:5400359,afterok:5400360
5400363
--dependency=afterok:5400357,afterok:5400358,afterok:5400359,afterok:5400360
5400364
afterok:5400361,afterok:5400362,afterok:5400363,afterok:5400364
Finally a single job that waits for the previous 4
5400361 5400362 5400363 5400364
5400365
+-+-+- REPORT OF PENDING JOB DEPENDECIES +-+-+-
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
           5400356     short    atest tkaiser2 PD       0:00      1 (None) 
           5400357     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400358     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400359     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400360     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400361     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400362     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400363     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400364     short    atest tkaiser2 PD       0:00      1 (Dependency) 
           5400365     short    atest tkaiser2 PD       0:00      1 (Dependency) 
job 5400356
   JobState=PENDING Reason=None Dependency=(null)
job 5400357
   JobState=PENDING Reason=Dependency Dependency=afterok:5400356(unfulfilled)
job 5400358
   JobState=PENDING Reason=Dependency Dependency=afterok:5400357(unfulfilled)
job 5400359
   JobState=PENDING Reason=Dependency Dependency=afterok:5400358(unfulfilled)
job 5400360
   JobState=PENDING Reason=Dependency Dependency=afterok:5400359(unfulfilled)
job 5400361
   JobState=PENDING Reason=Dependency Dependency=afterok:5400357(unfulfilled),afterok:5400358(unfulfilled),afterok:5400359(unfulfilled),afterok:5400360(unfulfilled)
job 5400362
   JobState=PENDING Reason=Dependency Dependency=afterok:5400357(unfulfilled),afterok:5400358(unfulfilled),afterok:5400359(unfulfilled),afterok:5400360(unfulfilled)
job 5400363
   JobState=PENDING Reason=Dependency Dependency=afterok:5400357(unfulfilled),afterok:5400358(unfulfilled),afterok:5400359(unfulfilled),afterok:5400360(unfulfilled)
job 5400364
   JobState=PENDING Reason=Dependency Dependency=afterok:5400357(unfulfilled),afterok:5400358(unfulfilled),afterok:5400359(unfulfilled),afterok:5400360(unfulfilled)
job 5400365
   JobState=PENDING Reason=Dependency Dependency=afterok:5400361(unfulfilled),afterok:5400362(unfulfilled),afterok:5400363(unfulfilled),afterok:5400364(unfulfilled)
el2:collect> 


After all jobs run we have:

el2:collect> ls -d ser*
ser1  ser2  ser3  ser4  ser5
el2:collect> ls ser*
ser1:
5400356.out

ser2:
5400356.out  5400357.out

ser3:
5400356.out  5400357.out  5400358.out

ser4:
5400356.out  5400357.out  5400358.out  5400359.out

ser5:
5400356.out  5400357.out  5400358.out  5400359.out  5400360.out
el2:collect> 
el2:collect> 
el2:collect> ls -d par*
par1  par2  par3  par4
el2:collect> ls par*
par1:
5400356.out  5400357.out  5400358.out  5400359.out  5400360.out  5400361.out

par2:
5400356.out  5400357.out  5400358.out  5400359.out  5400360.out  5400362.out

par3:
5400356.out  5400357.out  5400358.out  5400359.out  5400360.out  5400363.out

par4:
5400356.out  5400357.out  5400358.out  5400359.out  5400360.out  5400364.out
el2:collect> 
el2:collect> 
el2:collect> ls -d final
final
el2:collect> ls final
5400361.out  5400362.out  5400363.out  5400364.out  5400365.out
el2:collect> 


++++



