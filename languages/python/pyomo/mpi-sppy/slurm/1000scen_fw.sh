#!/bin/bash
#SBATCH --nodes=223             # Number of nodes
#SBATCH --ntasks=4000           # Request 4000 CPU cores
#SBATCH --time=00:10:00         # Job should run for up to 10 minutes
#SBATCH --account=aces  	# Where to charge NREL Hours
#SBATCH --mail-user=Firstname.Lastname@nrel.gov  # If you want email notifications
#SBATCH --mail-type=BEGIN,END,FAIL		 # When you want email notifications
#SBATCH --output=1000scen_fw.%j.out  # %j will be replaced with the job ID

module load conda               # load conda
module load xpressmp            # load xpress

conda activate pyomo

cd ${HOME}/software/mpi-sppy/paperruns/larger_uc

srun python -m mpi4py uc_cylinders.py --bundles-per-rank=0 --max-iterations=100 --default-rho=1.0 --num-scens=1000 --max-solver-threads=2 --solver-name=xpress_persistent --rel-gap=0.00001 --abs-gap=1 --no-cross-scenario-cuts
