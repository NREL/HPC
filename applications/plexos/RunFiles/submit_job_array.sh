#!/bin/bash
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --ntasks-per-node=104 # Tasks per node to be run
#SBATCH --mail-type=ALL # Get emails for everything
#SBATCH --job-name="PLXArray"
#SBATCH --out=%J.out
#SBATCH --error=%J.err

# export filename=FILENAME
# export LIST=model_list  # defaults to in_list
# sbatch -A account -p partition -t time --array=1-3 ./array.sh

###############################################################################
# Bash functions
###############################################################################

# This function attempts to ping the PLEXOS license server 5 times. If it is
# unable to find the server after 5 attempts, it exits with a status code 
# 404.
# NOTE: THIS DOES NOT CHECK IF YOUR LICENSE IS VALID!
ping_license_servers () {
    for attempt in `seq 5`
    do
        ping -c 2 10.60.3.188 > /dev/null
        if [ $? -eq 1 ] ; then
            date
            echo -e "Can not see license server. \nWill try again in 60 seconds.\n"
        else
            return 0
        fi
        if [ $attempt -lt 5 ] ; then sleep 60 ; fi
    done
    date
    echo "license lookup failed - exiting"
    date >> license.fail
    hostname >> license.fail
    ping -c 2 10.60.3.188 >> license.fail

  exit 404
}

###############################################################################
# This is where the actual job execution happens
###############################################################################

ping_license_servers
if [ $? -eq 404 ] ; then
    echo Could not find the license server
else
    # Load the appropriate modules
    module purge
    module load plexos/9.200R06
    
    cd /scratch/${USER}/HPC/applications/plexos/RunFiles/

    # get the JOB and SUBJOB ID
    if [[ $SLURM_ARRAY_JOB_ID ]] ; then
        export JOB_ID=$SLURM_ARRAY_JOB_ID
        export SUB_ID=$SLURM_ARRAY_TASK_ID
    else
        export JOB_ID=$SLURM_JOB_ID
        export SUB_ID=1
    fi

    # Check if a list of models has been provided in environment variables
    if [ -z ${LIST+x} ]; then 
        echo "LIST is unset"
        export LIST=in_list
    else 
        echo "LIST is set to '$LIST'"
    fi

    # Export all other environment variables to be used in the script
    export MAX_TEMP_FILE_AGE=50 # Environment variable used by PLEXOS to tell how long (in days) before deleting the file
    export PLEXOS_TEMP=/scratch/$USER/tmp/$JOBID
    export TEMP=$PLEXOS_TEMP

    # make a top level directory for the job if it does not already exist and enter it
    mkdir -p $JOB_ID

    export model=`head -n $SUB_ID $LIST | tail -1`

    # Finally run the PLEXOS model
    $PLEXOS/PLEXOS64 -n ${filename}.xml -m ${model} -o ${JOB_ID} -cu nrelplexos -cp Nr3lplex0s

fi
