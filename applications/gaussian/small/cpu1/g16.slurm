#!/bin/bash
#SBATCH --time=4:00:00             
#SBATCH --nodes=1
#SBATCH --job-name=s.cpu1
#SBATCH --output=std.out
#SBATCH --error=std.err
#SBATCH --account=hpcapps
#SBATCH --gres=gpu:2

# Load Gaussian module to set environment
module purge
module use /nopt/nrel/apps/modules/centos77/modulefiles/
module load gaussian/G16C_GPU
module list

cd $SLURM_SUBMIT_DIR

# Set script variables
INPUT_BASENAME=G16_test
INPUT_FILE=$INPUT_BASENAME.com
GAUSSIAN_EXEC=g16
MEMSIZE=5GB 
SCRATCH=/tmp/scratch/$SLURM_JOB_ID
SCRATCH2=/dev/shm 
# 
# Check on editing input file. If scratch directories 
# are listed then file is used un-changed, if 3-line 
# header not present, then script prepends these lines 
# to the input file to be used in execution line 
# 
NUMRWFLINES=`grep "RWF" $INPUT_FILE | wc -l` 
if [ $NUMRWFLINES -eq 1 ]; then 
 echo "standard file found" 
 cp $INPUT_FILE infile 
else 
 echo "prepending lines to input file" 
 echo "%RWF=$SCRATCH2/,$MEMSIZE,$SCRATCH/,-1" > infile 
 echo "%NoSave" >> infile 
 echo " " >> infile 
 cat $INPUT_FILE >> infile 
fi 

# 
# Run gaussian Peregrine script (performs much of the Gaussian setup) 
g16_eagle 

# 
# Set required Gaussian environment variables 
# 
if [ $SLURM_JOB_NUM_NODES -gt 1 ]; then 
 export GAUSS_LFLAGS='-vv -opt "Tsnet.Node.lindarsharg: ssh"' 
 export GAUSS_EXEDIR=$g16root/g16/linda-exe:$GAUSS_EXEDIR 
fi 
export GAUSS_SCRDIR=$SCRATCH2 
# 
# Gaussian needs scratch directories 
# Note: sometimes files may have been left behind in 
# on-node memory by other jobs that terminated incorrectly 
# so clean these to make sure there is enough space. 
# 
 
mkdir $SCRATCH 
rm -rf $SCRATCH2/* 

# Run Gaussian job 
$GAUSSIAN_EXEC < infile >& $INPUT_BASENAME.log 
rm infile

rm $SCRATCH/*
rmdir $SCRATCH

