#!/bin/bash

module load singularity-container

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to start_spark_worker.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath $1)

overhead_memory_kb=$2
spark_cluster=$3
if [ -z ${overhead_memory_kb} ]; then
    echo "Error: overhead_memory_kb must be passed to start_spark_worker.sh"
    exit 1
fi
if [ -z ${spark_cluster} ]; then
    echo "Error: spark_cluster must be passed to start_spark_worker.sh"
    exit 1
fi

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh
total_memory_kb=$(get_memory_kb)
memory_kb=$((${total_memory_kb} - ${overhead_memory_kb}))
echo "Start worker on `hostname` with memory ${memory_kb}k URL=${spark_cluster}"
exec_spark_process start-worker.sh -m ${memory_kb}k ${spark_cluster}
