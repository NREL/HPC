#!/bin/bash

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to stop_container.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

echo "Stop apptainer instance on $(hostname)"

module load ${CONTAINER_MODULE}
apptainer exec instance://${CONTAINER_NAME} stop-worker.sh
apptainer instance stop ${CONTAINER_NAME}

enable_pg=$(get_config_variable "enable_postgres_metastore")
if ${enable_pg}; then
    apptainer exec instance://pg-server pg_ctl stop
    apptainer instance stop pg-server
fi
