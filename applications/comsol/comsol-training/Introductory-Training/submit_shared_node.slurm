#!/bin/bash
#SBATCH --job-name=comsol-batch-1proc
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --account=hpcapps
#SBATCH --partition=shared
#SBATCH --output=comsol-%j.out
#SBATCH --error=comsol-%j.err

# This helps ensure your job runs from the directory
# from which you ran the sbatch command
cd $SLURM_SUBMIT_DIR

# Set up environment, and list to stdout for verification
module purge
module load comsol

inputfile=$SLURM_SUBMIT_DIR/myinputfile.mph
outputfile=$SLURM_SUBMIT_DIR/myoutputfilename

# Run a COMSOL job with 104 threads.
# -np = number of threads per rank

comsol batch -np 13 -inputfile $inputfile -outputfile 

