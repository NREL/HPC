#!/bin/bash

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

if [ -z ${1} ]; then
    error " CONFIG_DIR must be passed to record_resources.sh"
fi
export CONFIG_DIR=$(realpath $1)
memory_gb=$(get_memory_gb)
num_cpus=$(get_num_cpus)
echo "${memory_gb}" >> ${CONFIG_DIR}/conf/worker_memory
echo "${num_cpus}" >> ${CONFIG_DIR}/conf/worker_num_cpus
