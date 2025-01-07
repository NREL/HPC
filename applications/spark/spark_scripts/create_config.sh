#!/bin/bash

# Creates the base config file and copies the Spark configuration files to the user's directory.

CONTAINER_PATH="/datasets/images/apache_spark/spark354_py311.sif"
CONTAINER_NAME="spark"
DIRECTORY=$(pwd)
NODE_MEMORY_OVERHEAD_GB=10
ENABLE_DERBY_METASTORE=false
ENABLE_POSTGRES_METASTORE=false
POSTGRES_PASSWORD="postgres"
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
  -S, --hive-metastore                Create a Hive metastore with Spark defaults (Apache Derby).
                                      Supports only one Spark session. [Default: false]
  -p, --postgres-hive-metastore       Create a metastore with PostgreSQL.
                                      Supports multiple Spark sessions. [Default: false]
  -P, --postgres-password TEXT         Password for PostgreSQL. [Default: random string]
  -s, --metastore-dir TEXT             Set a custom directory for the metastore and warehouse. [Default: current]
  -t, --thrift-server                  Enable the Thrift server to connect a SQL client. [Default: false]

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
    -S|--hive-metastore)
      ENABLE_DERBY_METASTORE=true
      shift
      ;;
    -p|--postgres-hive-metastore)
      ENABLE_POSTGRES_METASTORE=true
      shift
      ;;
    -P|--postgres-password)
      POSTGRES_PASSWORD=$2
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

if [ ${POSTGRES_PASSWORD} == "postgres" ]; then
    POSTGRES_PASSWORD=$(openssl rand -base64 10 | tr -dc 'a-zA-Z0-9' | head -c 10)
fi

echo "container = ${CONTAINER_PATH}" > ${CONFIG_FILE}
echo "container_instance_name = ${CONTAINER_NAME}" >> ${CONFIG_FILE}
echo "node_memory_overhead_gb = ${NODE_MEMORY_OVERHEAD_GB}" >> ${CONFIG_FILE}
echo "enable_derby_metastore = ${ENABLE_DERBY_METASTORE}" >> ${CONFIG_FILE}
echo "enable_postgres_metastore = ${ENABLE_POSTGRES_METASTORE}" >> ${CONFIG_FILE}
echo "postgres_password = ${POSTGRES_PASSWORD}" >> ${CONFIG_FILE}
echo "postgres_data_dir = ${DIRECTORY}/pg-data" >> ${CONFIG_FILE}
echo "postgres_run_dir = ${DIRECTORY}/pg-run" >> ${CONFIG_FILE}
echo "metastore_dir = ${METASTORE_DIR}" >> ${CONFIG_FILE}
echo "thrift_server = ${ENABLE_THRIFT_SERVER}" >> ${CONFIG_FILE}

echo "Created configuration files in ${DIRECTORY}/conf with config file ${CONFIG_FILE}"
