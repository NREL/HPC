module purge
module load centos mono/4.6.2.7 xpressmp/8.0.4 plexos/7.400.2 conda
export PYTHONPATH=/nopt/nrel/apps/plexos/plexos-coad
export PLEXOS_TEMP=/scratch/$USER/tmp/$PBS_JOBID
export TEMP=$PLEXOS_TEMP
mkdir -p $PLEXOS_TEMP
