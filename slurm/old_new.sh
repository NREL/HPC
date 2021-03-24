#!/bin/bash
#SBATCH --job-name="atest"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:05:00
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j

:<<++++

Author: Tim Kaiser

This script shows how to use slurm dependencies to build
complex workflows.  See FAN for a complete description.


++++

cd $SLURM_SUBMIT_DIR
module purge
# needed for threaded apps built with Intel compilers
module load comp-intel
module load mpt

# Make a directory for this run and go there.
# If NEW_DIR is defined then we use that for
# our directory name or we set it to SLURM_JOBID.

if [ -z "$NEW_DIR" ]  ; then
  export NEW_DIR=$SLURM_JOBID
fi
mkdir $NEW_DIR

# If we have OLD_DIR defined then we copy old to new.
if [ -n "$OLD_DIR" ]  ; then
  cp $OLD_DIR/* $NEW_DIR
fi

# If we have OLD_FILES defined then we copy files.
# This copies a single output file from a set of
# directories instead of the whole directory.
if [ -n "$OLD_FILES" ]  ; then
  for afile in $OLD_FILES ; do 
    cp */$afile.out $NEW_DIR
  done
fi

cd $NEW_DIR
export OMP_NUM_THREADS=2
# Here we just run the hello world program “phostname"
srun -n 8 $SLURM_SUBMIT_DIR/phostone -F -t 10 -T >$SLURM_JOBID.out

