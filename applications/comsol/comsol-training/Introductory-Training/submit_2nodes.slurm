#!/bin/bash
#SBATCH --job-name=comsol-batch-4proc
#SBATCH --time=00:20:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --partition=debug
#SBATCH --account=hpcapps
#SBATCH --reservation=phase2-maintenance
#SBATCH --output=comsol-%j.out
#SBATCH --error=comsol-%j.err

# This helps ensure your job runs from the directory
# from which you ran the sbatch command
cd $SLURM_SUBMIT_DIR

# Set up environment, and list to stdout for verification
module purge
module load comsol


inputfile=/projects/hpcapps/comsol_model/beam_model.mph
outputfile=/projects/hpcapps/comsol_model/beam_model_results.mph


# Run a 2-node, 4-rank parallel COMSOL job with 18 threads for each rank.
# -nn = total number of MPI ranks
# -nnhost = number of MPI ranks per host
# -np = number of threads per rank

comsol –nn 6 -nnhost 2 batch -np 52 -inputfile $inputfile -outputfile 
