module use /nopt/nrel/apps/modules/candidate/modulefiles
module purge
module load epel gcc mono/3.2.3 xpressmp/7.8.0 plexos/6.400.2
module load conda
module load coad
export PLEXOS_TEMP=/scratch/$USER/tmp/$PBS_JOBID
export TEMP=$PLEXOS_TEMP
mkdir -p $PLEXOS_TEMP
