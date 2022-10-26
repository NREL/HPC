#!/bin/bash

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to stop_spark_cluster.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

module load singularity-container
singularity exec instance://${CONTAINER_INSTANCE_NAME} stop-history-server.sh
singularity exec instance://${CONTAINER_INSTANCE_NAME} stop-all.sh
for node_name in $(cat ${CONFIG_DIR}/conf/workers); do
    ssh ${USER}@${node_name} ${SCRIPT_DIR}/stop_container.sh ${CONFIG_DIR}
done

echo "Stopped all Spark processes and containers on all nodes."

