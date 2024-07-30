#!/bin/bash

# exit when a bash command fails
set -e

unset XDF_RUNTIME_DIR

# Use full CPU node script by default. For a shared partition or GPU node, comment out the next line and uncomment the line corresponding to the script you would like to run.
RES=$(sbatch sbatch_jupyter.sh)
# RES=$(sbatch shared_sbatch_jupyter.sh)
# RES=$(sbatch gpu_sbatch_jupyter.sh) 

jobid=${RES##* }

tries=1
wait=1
echo "Checking job status.."
while :
do
        status=$(scontrol show job $jobid | grep JobState | awk '{print $1}' | awk -F= '{print $2}')
        if [ $status == "RUNNING" ]
        then
                echo "job is running!"
                echo "getting jupyter information, hang tight.."
                while :
                do
                        if [ ! -f slurm-$jobid.out ]
                        then
                                echo "waiting for slurm output to be written"
                                let "wait+=1"
                                sleep 1s
                        elif [ $wait -gt 120 ]
                        then
                                echo "timed out waiting for output from job."
                                echo "check to make sure job didn't fail"
                                exit 0
                        else
                                check=$(cat slurm-$jobid.out | grep http://127.0.0.1 | wc -l)
                                if [ $check -gt 0 ]
                                then
                                        echo "okay, now run the follwing on your local machine:"
                                        echo $(cat slurm-$jobid.out | grep ssh)
                                        echo "then, navigate to the following on your local browser:"
                                        echo $(cat slurm-$jobid.out | grep http://127.0.0.1 | head -1 | awk {'print $5'})
                                        exit 0
                                else
                                        let "wait+=1"
                                        sleep 1s
                                fi
                        fi
                done
                exit 0
        elif [ $tries -gt 120 ]
        then
                echo "timeout.. terminating job."
                scancel $jobid
                exit 0
        else
                echo "job still pending.."
                sleep 10s
        fi
        ((tries++))
done
