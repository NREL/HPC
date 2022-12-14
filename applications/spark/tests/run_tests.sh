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
sbatch batch_job.sh
echo "Check results manually when complete and delete the directory ./run."
