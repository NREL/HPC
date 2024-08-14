#!/bin/bash

if [ -z $1 ]; then
    echo "Error: at least one SLURM job ID must be provided"
    exit 1
fi

slurm_job_ids=$@
nodes=()
for job_id in ${slurm_job_ids}
do
    # The squeue command will produce multiple lines if the job is heterogeneous.
    for host_list in $(squeue -j ${job_id} --format="%5D %1000N" -h | awk '{print $2}')
    do
        host_names=$(scontrol show hostnames ${host_list})
        for host_name in $host_names
        do
            nodes+=($host_name)
        done
    done
done

echo "${nodes[@]}"
