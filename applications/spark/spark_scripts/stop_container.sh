#!/bin/bash

module load singularity-container

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to stop_container.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh
echo "Stop singularity instance on $(hostname)"

singularity instance stop ${CONTAINER_INSTANCE_NAME}
