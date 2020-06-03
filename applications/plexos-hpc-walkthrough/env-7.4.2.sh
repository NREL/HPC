module use /nopt/nrel/apps/modules/candidate/modulefiles
module purge
module load epel gcc/4.8.2 mono/4.6.2.7 xpressmp plexos/7.400.2 conda coad
export PLEXOS_TEMP=/scratch/$USER/tmp/$PBS_JOBID
export TEMP=$PLEXOS_TEMP
mkdir -p $PLEXOS_TEMP
