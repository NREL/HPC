#!/bin/bash

# Creates the base config file and copies the Spark configuration files to the user's directory.

CONTAINER_PATH="/datasets/images/apache_spark/spark352_py311.sif"
CONTAINER_NAME="spark"
DIRECTORY=$(pwd)
NODE_MEMORY_OVERHEAD_GB=10
METASTORE_DIR="."
ENABLE_THRIFT_SERVER=false
ARGS=()

read -r -d "" USAGE << EOM
Usage: $(basename $0) [OPTIONS]...

  Create base configuration files.

Options:
  -C, --container-name TEXT            Apptainer instance name [default: ${CONTAINER_NAME}]
  -c, --container-path TEXT            Apptainer container path [default: ${CONTAINER_PATH}]
  -d, --directory TEXT                 Base config directory [default: current]
  -o, --node-memory-overhead-gb INTEGER
                                       Memory to reserve for system processes. [Default: ${NODE_MEMORY_OVERHEAD_GB}]
  -s, --metastore-dir TEXT             Set a custom directory for the metastore and warehouse. [Default: current]
  -t, --thrift-server TEXT             Enable the Thrift server to connect a SQL client. [Default: false]

Example:
  $(basename $0) -c /${HOME}/my_container.sif
EOM

while [[ $# -gt 0 ]]; do
  case $1 in
    -C|--container-name)
      CONTAINER_NAME=$2
      shift
      shift
      ;;
    -c|--container-path)
      CONTAINER_PATH=$2
      shift
      shift
      ;;
    -d|--directory)
      DIRECTORY=$2
      shift
      shift
      ;;
    -o|--node-memory-overhead-gb)
      NODE_MEMORY_OVERHEAD_GB=$2
      shift
      shift
      ;;
    -s|--metastore-dir)
      METASTORE_DIR=$2
      shift
      shift
      ;;
    -t|--thrift-server)
      ENABLE_THRIFT_SERVER=true
      shift
      ;;
    -h|--help)
      echo "${USAGE}"
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

if ! [ -d ${DIRECTORY} ]; then
    echo "Error: ${DIRECTORY} does not exist"
    exit 1
fi

if ! [ -f ${CONTAINER_PATH} ]; then
    echo "Error: container_path=${CONTAINER_PATH} does not exist"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONF_DIR=$(dirname ${SCRIPT_DIR})/conf
if [ -z ${CONF_DIR} ]; then
    echo "Error: the directory ${CONF_DIR} does not exist"
    exit 1
fi
set -e
cp -r ${CONF_DIR} ${DIRECTORY}
CONFIG_FILE="${DIRECTORY}/config"

echo "container = ${CONTAINER_PATH}" > ${CONFIG_FILE}
echo "container_instance_name = ${CONTAINER_NAME}" >> ${CONFIG_FILE}
echo "node_memory_overhead_gb = ${NODE_MEMORY_OVERHEAD_GB}" >> ${CONFIG_FILE}
echo "metastore_dir = ${METASTORE_DIR}" >> ${CONFIG_FILE}
echo "thrift_server = ${ENABLE_THRIFT_SERVER}" >> ${CONFIG_FILE}

echo "Created configuration files in ${DIRECTORY}/conf with config file ${CONFIG_FILE}"
