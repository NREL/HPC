#!/bin/bash

CONFIG_DIR=$(pwd)
CONTAINER_PATH="/datasets/images/apache_spark/spark352_py311.sif"
CONTAINER_NAME="spark"
NODE_MEMORY_OVERHEAD_GB=10
DRIVER_MEMORY_GB=10
ENABLE_DYNAMIC_ALLOCATION=false
ENABLE_HISTORY_SERVER=false
METASTORE_DIR="."
ENABLE_THRIFT_SERVER=false
# Many online docs say executors max out with 5 threads.
EXECUTOR_CORES=5
PARTITION_MULTIPLIER=1
SLURM_JOB_IDS=()
SPARK_SCRATCH="tmpfs"

# Main

read -r -d "" USAGE << EOM
Usage: $(basename $0) [OPTIONS]... [SLURM_JOB_ID]...

  Configure settings in spark-defaults.conf based on hardware resources in one or more SLURM job IDs.
  Reads SLURM_JOB_ID from the environment.

Options:
  -C, --container-name TEXT           Apptainer instance name [default: ${CONTAINER_NAME}]
  -c, --container-path TEXT           Apptainer container path [default: ${CONTAINER_PATH}]
  -d, --directory TEXT                Base config directory [default: current]
  -o, --node-memory-overhead-gb INTEGER
                                      Memory to reserve for system processes. [Default: ${NODE_MEMORY_OVERHEAD_GB}]
  -D, --dynamic-allocation            Enable dynamic resource allocation. [Default: false]
  -H, --history-server                Enable the history server. [Default: false]
  -M, --driver-memory-gb INTEGER      Driver memory in GB. [Default: ${DRIVER_MEMORY_GB}]
  -e, --executor-cores INTEGER        Number of cores per executor. [Default: ${EXECUTOR_CORES}]
  -l, --spark-scratch TEXT            Directory given to Spark workers for shuffle writes and log files.
                                      [Default: compute node tmpfs]
  -m, --partition-multiplier INTEGER  Set spark.sql.shuffle.partitions to number of
                                      cores multiplied by this value. [Default: ${PARTITION_MULTIPLIER}]
  -s, --metastore-dir TEXT            Set a custom directory for the metastore and warehouse. [Default: current]
  -t, --thrift-server TEXT            Enable the Thrift server to connect a SQL client. [Default: false]

Example:
  $(basename $0) --driver-memory-gb 2
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
      CONFIG_DIR=$2
      shift
      shift
      ;;
    -o|--node-memory-overhead-gb)
      NODE_MEMORY_OVERHEAD_GB=$2
      shift
      shift
      ;;
    -D|--dynamic-allocation)
      ENABLE_DYNAMIC_ALLOCATION=true
      shift
      ;;
    -H|--history-server)
      ENABLE_HISTORY_SERVER=true
      shift
      ;;
    -M|--driver-memory-gb)
      DRIVER_MEMORY_GB=${2}
      shift
      shift
      ;;
    -e|--executor-cores)
      EXECUTOR_CORES=${2}
      shift
      shift
      ;;
    -l|--spark-scratch)
      SPARK_SCRATCH=${2}
      shift
      shift
      ;;
    -m|--shuffle-partitions-multiplier)
      PARTITION_MULTIPLIER=${2}
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
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      SLURM_JOB_IDS+=("$1")
      shift
      ;;
  esac
done

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
set -e
create_flags=""
if [[ ${METASTORE_DIR} != "." ]]; then
    create_flags+=" -s ${METASTORE_DIR}"
fi
if [ ${ENABLE_THRIFT_SERVER} = true ]; then
    create_flags+=" -t"
fi

bash ${script_dir}/create_config.sh \
    ${create_flags} \
    -C ${CONTAINER_NAME} \
    -c ${CONTAINER_PATH} \
    -d ${CONFIG_DIR} \
    -o ${NODE_MEMORY_OVERHEAD_GB}

. ${script_dir}/common.sh
if [ ${ENABLE_DYNAMIC_ALLOCATION} = true ]; then
    flags="-D"
else
    flags=""
fi
if [ ${ENABLE_HISTORY_SERVER} = true ]; then
    flags+=" -H"
fi

bash ${script_dir}/configure_spark.sh \
    ${flags} \
    -M ${DRIVER_MEMORY_GB} \
    -d ${CONFIG_DIR} \
    -e ${EXECUTOR_CORES} \
    -l ${SPARK_SCRATCH} \
    -m ${PARTITION_MULTIPLIER} \
    ${SLURM_JOB_IDS[*]}

bash ${script_dir}/start_spark_cluster.sh -d ${CONFIG_DIR} ${SLURM_JOB_IDS[*]}
