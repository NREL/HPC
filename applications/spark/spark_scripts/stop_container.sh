#!/bin/bash

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to stop_container.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

echo "Stop ${CONTAINER_EXEC} instance on $(hostname)"

module load ${CONTAINER_MODULE}
${CONTAINER_EXEC} exec instance://${CONTAINER_NAME} stop-worker.sh
${CONTAINER_EXEC} instance stop ${CONTAINER_NAME}
