#!/bin/bash

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to start_container.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

module load ${CONTAINER_MODULE}

echo "Start ${CONTAINER_EXEC} instance on $(hostname)"
${CONTAINER_EXEC} instance start \
    -B $(mktemp -d ${CONFIG_DIR}/run/`hostname`_XXXX):/run \
    -B ${CONFIG_DIR}/dropbear/:/etc/dropbear \
    ${LUSTRE_BIND_MOUNTS} \
    $(get_spark_bind_mounts ${CONFIG_DIR}) \
    ${CONTAINER_PATH} \
    ${CONTAINER_NAME}
if [ $? -ne 0 ]; then
    exit 1
fi
${CONTAINER_EXEC} exec instance://${CONTAINER_NAME} service dropbear start
