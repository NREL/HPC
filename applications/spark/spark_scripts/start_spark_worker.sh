#!/bin/bash

# This function allows each worker to find a writable scratch directory.
# Background:
#   - Kestrel compute nodes that have local storage mount that drive at /tmp/scratch.
#     However, that directory is not writable.
#   - At Slurm job allocation time, the HPC scripts create /tmp/scratch/<job-id>, which is
#     writable. Those scripts export the environment variable TMPDIR with that value, but this
#     worker script won't have access to it because it runs in an ssh session.
#   - We may be creating a Spark cluster that spans multiple Slurm jobs, and so there is no way
#     for the parent script to set this correctly for each node ahead of time.
function get_scratch_dir()
{
    base_scratch="/tmp/scratch"
    if ! [ -d ${base_scratch} ]; then
        # No local storage
        echo "/tmp"
        return
    fi
    for x in $(ls ${base_scratch}); do
        path="${base_scratch}/${x}"
        echo ${x} | grep "^[0-9]\+$" > /dev/null
        if [ $? -eq 0 ] && [ -w ${path} ]; then
            echo "${path}"
            return
        fi
    done
}

export CONFIG_DIR=$(realpath $1)
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

module load ${CONTAINER_MODULE}

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to start_spark_worker.sh"
    exit 1
fi

memory_gb=$2
spark_cluster=$3
if [ -z ${memory_gb} ]; then
    echo "Error: memory_gb must be passed to start_spark_worker.sh"
    exit 1
fi
if [ -z ${spark_cluster} ]; then
    echo "Error: spark_cluster must be passed to start_spark_worker.sh"
    exit 1
fi

spark_env=${CONFIG_DIR}/conf/spark-env.sh
if ! [ -f ${spark_env} ]; then
    echo "spark-env.sh does not exist at ${spark_env}"
    exit 1
fi

grep "^\s*SPARK_LOCAL_DIRS" ${spark_env} > /dev/null
if [ $? -ne 0 ]; then
    scratch_dir=$(get_scratch_dir)
    export SPARK_LOCAL_DIRS=${scratch_dir}/spark/local
fi

grep "^\s*SPARK_WORKER_DIR" ${spark_env} > /dev/null
if [ $? -ne 0 ]; then
    scratch_dir=$(get_scratch_dir)
    export SPARK_WORKER_DIR=${scratch_dir}/spark/worker
fi

echo "Start worker on $(hostname) with memory ${memory_gb}g local_dirs=${SPARK_LOCAL_DIRS} worker_dir=${SPARK_WORKER_DIR} URL=${spark_cluster}"
exec_spark_process start-worker.sh -m ${memory_gb}g ${spark_cluster}
