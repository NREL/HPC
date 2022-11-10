#!/bin/bash

CONFIG_DIR=$(pwd)
ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--directory)
      CONFIG_DIR=$(realpath ${2})
      shift
      shift
      ;;
    -h|--help)
      echo "Usage: $(basename $0) [-d|--directory CONFIG_DIRECTORY]"
      exit 0
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if ! [ -d ${CONFIG_DIR} ]; then
    echo "Error: CONFIG_DIR=${CONFIG_DIR} does not exist"
    exit 1
fi

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

module load singularity-container
# Checking for errors is not necessary.
singularity exec instance://${CONTAINER_INSTANCE_NAME} stop-history-server.sh

# Something about the ssh configuration causes a warning when the Spark
# scripts ssh to each worker node. It doesn't happen in our ssh commands.
# Workaround the issue by stopping the Spark worker inside stop_container.sh.
# singularity exec instance://${CONTAINER_INSTANCE_NAME} stop-all.sh
singularity exec instance://${CONTAINER_INSTANCE_NAME} stop-master.sh
for node_name in $(cat ${CONFIG_DIR}/conf/workers); do
    ssh ${USER}@${node_name} ${SCRIPT_DIR}/stop_container.sh ${CONFIG_DIR}
done

echo "Stopped all Spark processes and containers on all nodes."
