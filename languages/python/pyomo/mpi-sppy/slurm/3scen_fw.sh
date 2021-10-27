#!/bin/bash
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=12             # Request 12 CPU cores
#SBATCH --time=00:05:00         # Job should run for up to 5 minutes
#SBATCH --account=aces  	# Where to charge NREL Hours
#SBATCH --mail-user=Firstname.Lastname@nrel.gov  # If you want email notifications
#SBATCH --mail-type=BEGIN,END,FAIL		 # When you want email notifications
#SBATCH --output=3scen_fw.%j.out  # %j will be replaced with the job ID

module load conda
module load xpressmp

conda activate pyomo

export MPICH_ASYNC_PROGRESS=1

cd ${HOME}/software/mpi-sppy/paperruns/larger_uc

srun python -u -m mpi4py uc_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=1.0 --num-scens=3 --max-solver-threads=2 --solver-name=xpress_persistent --rel-gap=0.00001 --abs-gap=1 --no-cross-scenario-cuts
