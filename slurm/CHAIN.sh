#!/bin/bash

:<<++++

Author: Tim Kaiser

This script shows how to use slurm dependencies to build
complex workflows.

(1) Starts with 5 jobs that need to run in sequence.

See FAN.sh for a complete descrption.

++++

# Here is the script we will run
export SCRIPT=old_new.sh

if [ -z ${1+x} ]; then
	echo USAGE:
	echo $0 account
	echo Your account needs to be set on the command line
	exit
fi
export ACC=$1

unset OLD_DIR
export NEW_DIR=job1
jid=`sbatch -J slurm_test -A $ACC $SCRIPT | awk '{print $NF }'`
echo $jid
 
for job in job2 job3 job4 job5 ; do
  export OLD_DIR=$NEW_DIR
  export NEW_DIR=$job
  jid=`sbatch -J slurm_test -A $ACC --dependency=afterok:$jid $SCRIPT | awk '{print $NF }'`
  echo $jid
done

