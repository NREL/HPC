#!/bin/bash

account=$1
if [ -z ${account} ]; then
    echo "Usage: $(basename $0) ACCOUNT"
    exit 1
fi

if [ -d run ]; then
    echo "The directory ./run already exists. Please delete it and re-run."
    exit 1
fi
mkdir run
cp batch_job.sh.template run/batch_job.sh
sed -i s/TEST_ACCOUNT/${account}/ run/batch_job.sh
if [ $? -ne 0 ]; then
    echo "Failed to set account"
    exit 1
fi

cd run
output=$(sbatch batch_job.sh)
echo "${output}"
job_id=$(echo "${output}" | grep -oE "[0-9]+")
echo "Monitor for job completion with squeue -j ${job_id}"
echo "When complete, check results by verifying ExitCode=0 with sacct -j ${job_id}"
echo "Delete the directory ./run to clean up."
