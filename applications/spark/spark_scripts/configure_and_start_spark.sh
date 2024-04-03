#!/bin/bash

CONFIG_DIR=$(pwd)
if [ ! -z ${NREL_CLUSTER} ] && [ ${NREL_CLUSTER} == "kestrel" ]; then
    CONTAINER_PATH="/kfs2/pdatasets/images/apache_spark/spark350_py311.sif"
else
    CONTAINER_PATH="/datasets/images/apache_spark/spark350_py311.sif"
fi
CONTAINER_NAME="spark"
NODE_MEMORY_OVERHEAD_GB=5
DRIVER_MEMORY_GB=1
ENABLE_DYNAMIC_ALLOCATION=false
ENABLE_HISTORY_SERVER=false
# Many online docs say executors max out with 5 threads.
EXECUTOR_CORES=5
PARTITION_MULTIPLIER=1
SLURM_JOB_IDS=()

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
  -m, --partition-multiplier INTEGER  Set spark.sql.shuffle.partitions to number of
                                      cores multiplied by this value. [Default: ${PARTITION_MULTIPLIER}]

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
    -m|--shuffle-partitions-multiplier)
      PARTITION_MULTIPLIER=${2}
      shift
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
bash ${script_dir}/create_config.sh \
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
    -m ${PARTITION_MULTIPLIER} \
    ${SLURM_JOB_IDS[*]}

bash ${script_dir}/start_spark_cluster.sh -d ${CONFIG_DIR} ${SLURM_JOB_IDS[*]}
