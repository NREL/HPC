#!/bin/bash

function setup()
{
    if ! [ -d dropbear ]; then
        singularity exec ${LUSTRE_BIND_MOUNTS} ${CONTAINER} ${SCRIPT_DIR}/make_dropbear.sh
    fi
    rm -rf ${CONFIG_DIR}/events && mkdir ${CONFIG_DIR}/events
    rm -rf ${CONFIG_DIR}/logs && mkdir ${CONFIG_DIR}/logs
    rm -rf ${CONFIG_DIR}/run && mkdir ${CONFIG_DIR}/run
    mkdir -p ${CONFIG_DIR}/conf
}

function write_worker_nodes()
{
    workers_file="${CONFIG_DIR}/conf/workers"
    rm -f ${workers_file}
    touch ${workers_file}

    for node in $(${SCRIPT_DIR}/get_node_names.sh ${SLURM_JOB_IDS})
    do
        echo "${node}" >> $workers_file
    done
}

function start_containers()
{
    for node_name in $(cat ${CONFIG_DIR}/conf/workers); do
        ssh ${USER}@${node_name} ${SCRIPT_DIR}/start_container.sh ${CONFIG_DIR}
    done

    echo "Started containers on all nodes"
}

function start_spark_processes()
{
    master_overhead_memory_kb=$((${MASTER_NODE_MEMORY_OVERHEAD_GB} * 1024 * 1024))
    worker_overhead_memory_kb=$((${WORKER_NODE_MEMORY_OVERHEAD_GB} * 1024 * 1024))
    master_node=$(hostname | tr -d '\n')
    spark_cluster=spark://${master_node}:7077

    exec_spark_process start-master.sh
    exec_spark_process start-history-server.sh
    ${SCRIPT_DIR}/start_spark_worker.sh ${CONFIG_DIR} ${master_overhead_memory_kb} ${spark_cluster}
    ret=$?
    if [[ $ret -ne 0 ]]; then
        echo "Error: Failed to start Spark worker on the master node: ${ret}"
        exit $ret
    fi
    echo "Started Spark processes on master node ${master_node}"

    # Spark does provide a way to start all nodes at once: start-workers.sh.
    # But that doesn't allow specifying memory for each node independently.
    for node_name in $(cat ${CONFIG_DIR}/conf/workers); do
        if [[ $node_name != ${master_node} ]]; then
            ssh ${USER}@${node_name} ${SCRIPT_DIR}/start_spark_worker.sh \
                ${CONFIG_DIR} ${worker_overhead_memory_kb} ${spark_cluster}
            ret=$?
            if [[ $ret -ne 0 ]]; then
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

module load singularity-container
setup
write_worker_nodes
start_containers
start_spark_processes
