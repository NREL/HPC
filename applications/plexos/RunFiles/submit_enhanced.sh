#!/bin/bash 
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --ntasks-per-node=104 # Tasks per node to be run
#SBATCH --time=00:30:00   # Required, estimate 5 minutes
#SBATCH --partition=debug
#SBATCH --mail-type=BEGIN,END,FAIL,REQUE
#SBATCH --job-name="PLSimple"

module purge
module load craype-x86-spr
module load gurobi/10.0.2 plexos/9.200R06

cd /scratch/${USER}/HPC/applications/plexos/RunFiles/

# We will make multiple attempts in case we cannot get a license
export WAIT_TIME=120
for attempt in {1..10}
do
    start_time=`date +%s`
    $PLEXOS/PLEXOS64 -n 5_bus_system_v2.xml -m 2024_yr_15percPV_MT_Gurobi -cu nrelplexos -cp Nr3lplex0s > fout_${SLURM_JOB_ID} 2>&1
    end_time=`date +%s`
    run_time=$((end_time-start_time))
    echo Run Time = ${run_time}
    grep "Unable to acquire license" fout_${SLURM_JOB_ID} >& /dev/null
    if [ $? -eq 0 ] ; then
		date                              >> ping.out
		ping -c 3 $HOSTNAME               >> ping.out
                ping -c 3 google.com              >> ping.out
                ping -c 3 10.60.3.188             >> ping.out
		mv fout_${SLURM_JOB_ID} fout_${SLURM_JOB_ID}_attempt_${attempt}
		echo will try again in $WAIT seconds
		sleep $WAIT
	else
		echo "found it"
		break
	fi
done