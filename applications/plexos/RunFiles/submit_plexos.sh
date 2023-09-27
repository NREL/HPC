#!/bin/bash
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --ntasks-per-node=104 # Tasks per node to be run
#SBATCH --mail-type=ALL # Get emails for everything
#SBATCH --job-name="PLMultiple"

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
    module load craype-x86-spr
    module load gurobi/10.0.2 plexos/9.200R06

    # Go to the correct project folder
    cd /scratch/${USER}/HPC/applications/plexos/RunFiles/
    # Finally run PLEXOS
    $PLEXOS/PLEXOS64 -n ${filename}.xml -m ${model} -cu nrelplexos -cp Nr3lplex0s > fout_${filename}_${model} 2>&1

    # Move the resulting output folder into the working directory.
fi
