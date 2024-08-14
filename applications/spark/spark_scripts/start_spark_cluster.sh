#!/bin/bash

function setup()
{
    if ! [ -d dropbear ]; then
        ${CONTAINER_EXEC} exec ${LUSTRE_BIND_MOUNTS} ${CONTAINER_PATH} ${SCRIPT_DIR}/make_dropbear.sh
    fi
    rm -rf ${CONFIG_DIR}/events && mkdir ${CONFIG_DIR}/events
    rm -rf ${CONFIG_DIR}/logs && mkdir ${CONFIG_DIR}/logs
    rm -rf ${CONFIG_DIR}/run && mkdir ${CONFIG_DIR}/run
    mkdir -p ${CONFIG_DIR}/conf
}

function start_containers()
{
    ${SCRIPT_DIR}/start_container.sh ${CONFIG_DIR}
    for node_name in $(cat ${CONFIG_DIR}/conf/workers); do
	if [ ${node_name} != $(hostname) ]; then
            ssh ${USER}@${node_name} ${SCRIPT_DIR}/start_container.sh ${CONFIG_DIR}
        fi
    done

    echo "Started containers on all nodes"
}

function start_spark_processes()
{
    master_node=$(hostname | tr -d '\n')
    spark_cluster=spark://${master_node}:7077
    node_memory_overhead_gb=$(get_node_memory_overhead_gb)

    exec_spark_process start-master.sh
    check_history_server_enabled
    if [ $? -eq 0 ]; then
        exec_spark_process start-history-server.sh
    fi
    enable_thrift_server=$(get_config_variable "thrift_server")
    if [ ${enable_thrift_server} == "true" ]; then
        exec_spark_process start-thriftserver.sh ${spark_cluster}
    fi
    echo "Started Spark master processes on ${master_node}"
    echo "Spark worker memory overhead = ${node_memory_overhead_gb} GB"
    worker_memory_gb=$(( $(get_worker_memory_gb) - ${node_memory_overhead_gb}))
    if is_heterogeneous_slurm_job; then
        echo "Don't start worker on master node."
    else
        ${SCRIPT_DIR}/start_spark_worker.sh ${CONFIG_DIR} ${worker_memory_gb} ${spark_cluster}
        ret=$?
        if [ $ret -ne 0 ]; then
            echo "Error: Failed to start Spark worker on the master node: ${ret}"
            exit $ret
        fi
        echo "Started Spark worker process on master node ${master_node}"
    fi

    for node_name in $(cat ${CONFIG_DIR}/conf/workers); do
        if [ $node_name != ${master_node} ]; then
            ssh ${USER}@${node_name} ${SCRIPT_DIR}/start_spark_worker.sh \
                ${CONFIG_DIR} ${worker_memory_gb} ${spark_cluster}
            ret=$?
            if [ $ret -ne 0 ]; then
                echo "Error: Failed to start the container on the worker node ${node_name}: ${ret}"
                exit $ret
            fi
            echo "Started Spark worker on worker node ${node_name}"
        fi
    done
}

# Main
CONFIG_DIR=$(pwd)
SLURM_JOB_IDS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--directory)
      CONFIG_DIR=$(realpath ${2})
      shift
      shift
      ;;
    -h|--help)
      echo "Usage: $(basename $0) [-d|--directory CONFIG_DIRECTORY] [SLURM_JOB_ID ...]"
      exit 0
      shift
      shift
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

if ! [ -d ${CONFIG_DIR} ]; then
    echo "Error: CONFIG_DIR=${CONFIG_DIR} does not exist"
    exit 1
fi

num_jobs=${#SLURM_JOB_IDS[@]}
if [ ${num_jobs} -eq 0 ]; then
    if ! [ -z ${SLURM_JOB_ID} ]; then
        SLURM_JOB_IDS+=${SLURM_JOB_ID}
    else
        echo "Error: at least one SLURM job ID must be passed"
        exit 1
    fi
fi

# Copied from
# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

# Behavioral notes
# There are two basic modes of operation.
#   1. The head node runs the Spark master process and a worker process.
#      The head node needs to account for CPU and memory for the master
#      process, Spark driver, and user application (Python, R, etc.).
#   2. The head node runs only a Spark master process.
#      For this to occur, the user must allocate a heterogeneous Slurm job
#      where the first group should be a single (ideally shared) node that
#      will be used for the master process.
#      This allows for a uniform worker configuration.
module load ${CONTAINER_MODULE}
run_checks
setup
start_containers
start_spark_processes

cat << EOM
###############################################################################

Run this command to use the Spark configuration:

  export SPARK_CONF_DIR=${CONFIG_DIR}/conf

###############################################################################
EOM
