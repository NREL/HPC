#!/bin/bash 
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --ntasks-per-node=104 # Tasks per node to be run
#SBATCH --time=00:10:00   # Required, estimate 5 minutes
#SBATCH --partition=debug
#SBATCH --mail-type=BEGIN,END,FAIL,REQUE
#SBATCH --job-name="PLSimple"

module purge
module load craype-x86-spr
module load gurobi/10.0.2 plexos/9.200R06

cd /scratch/${USER}/HPC/applications/plexos/RunFiles/

$PLEXOS/PLEXOS64 -n 5_bus_system_v2.xml -m 2024_yr_15percPV_Gurobi -cu nrelplexos -cp Nr3lplex0s > fout_${SLURM_JOB_ID} 2>&1