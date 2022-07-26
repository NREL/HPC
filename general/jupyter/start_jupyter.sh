#!/bin/bash
#SBATCH --job-name="jupyter"
#SBATCH --nodes=2
#SBATCH --account=hpcapps
#SBATCH --partition=debug
#SBATCH --time=01:00:00
##SBATCH --mail-type=ALL
##SBATCH --mail-user=tkaiser2@nrel.gov

#see ../../../slurm/source/jupyter.sh
#for a script to create the conda 
#environment referenced below
export MYVERSION=dompt
module load conda
source activate
source activate $MYVERSION
module load gcc/8.4.0 
module load mpt

date      > ~/jupyter.log
hostname >> ~/jupyter.log
jupyter notebook --NotebookApp.password='' --no-browser  >> ~/jupyter.log 2>&1


