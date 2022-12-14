#!/bin/bash

module load singularity-container

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to start_container.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh
echo "Start singularity instance on $(hostname)"
singularity instance start \
    -B $(mktemp -d ${CONFIG_DIR}/run/`hostname`_XXXX):/run \
    -B ${CONFIG_DIR}/dropbear/:/etc/dropbear \
    ${LUSTRE_BIND_MOUNTS} \
    $(get_spark_bind_mounts ${CONFIG_DIR}) \
    ${CONTAINER} \
    ${CONTAINER_INSTANCE_NAME}
singularity exec instance://${CONTAINER_INSTANCE_NAME} service dropbear start
