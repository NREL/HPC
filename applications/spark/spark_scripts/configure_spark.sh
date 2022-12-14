#!/bin/bash

CONFIG_DIR=$(pwd)
DRIVER_MEMORY_GB=1
ENABLE_DYNAMIC_ALLOCATION=false
ENABLE_HISTORY_SERVER=false
# Many online docs say executors max out with 5 threads.
EXECUTOR_CORES=5
PARTITION_MULTIPLIER=1
SLURM_JOB_IDS=()

# Check for errors in user input. Exit on error.
function check_errors()
{
    master_memory_gb=$(( 1 + ${DRIVER_MEMORY_GB} ))
    if [ ${ENABLE_HISTORY_SERVER} = true ]; then
        (( master_memory_gb += 1 ))
    fi

    if [ ${master_memory_gb} -gt ${MASTER_NODE_MEMORY_OVERHEAD_GB} ]; then
        error "master_node_memory_overhead_gb=${MASTER_NODE_MEMORY_OVERHEAD_GB} is too small." \
              "Increase it or reduce driver_memory_gb=${DRIVER_MEMORY_GB}"
    fi
}

# Configure executor settings in spark-defaults.conf.
function config_executors()
{
    rm -f ${CONFIG_DIR}/conf/worker_memory ${CONFIG_DIR}/conf/worker_num_cpus
    num_workers=0
    for node_name in $(${SCRIPT_DIR}/get_node_names.sh ${SLURM_JOB_IDS[@]}); do
        ssh ${USER}@${node_name} ${SCRIPT_DIR}/record_resources.sh ${CONFIG_DIR}
        ret=$?
        if [[ $ret -ne 0 ]]; then
            error "Failed to record resources on the worker node ${node_name}: ${ret}"
        fi
        (( num_workers += 1 ))
    done

    memory_gb_by_node=()
    lowest_memory_gb=0
    for node_mem in $(cat ${CONFIG_DIR}/conf/worker_memory); do
        mem=$(( ${node_mem} - ${MASTER_NODE_MEMORY_OVERHEAD_GB} ))
        if [ ${lowest_memory_gb} -eq 0 ] || [ ${node_mem} -lt ${lowest_memory_gb} ]; then
            lowest_memory_gb=${mem}
        fi
        memory_gb_by_node+=(${mem})
    done

    cpus_by_node=()
    lowest_num_cpus=0
    for node_num_cpus in $(cat ${CONFIG_DIR}/conf/worker_num_cpus); do
        # Leave one CPU for OS and management software.
        cpus=$(( ${node_num_cpus} - 1 ))
        if [ ${lowest_num_cpus} -eq 0 ] || [ ${cpus} -lt ${lowest_num_cpus} ]; then
            lowest_num_cpus=${cpus}
        fi
        cpus_by_node+=(${cpus})
    done

    min_executors_per_node=$(( ${lowest_num_cpus} / ${EXECUTOR_CORES} ))
    executor_memory_gb=$(( ${lowest_memory_gb} / ${min_executors_per_node} ))

    total_num_cpus=0
    total_num_executors=0
    for (( i=0; i < ${num_workers}; i++ )); do
        mem_gb=${memory_gb_by_node[${i}]}
        executors_by_mem=$(( ${mem_gb} / ${executor_memory_gb} ))
        cpus=${cpus_by_node[${i}]}
        executors_by_cpu=$(( ${cpus} / ${EXECUTOR_CORES} ))
        if [ ${executors_by_cpu} -le ${executors_by_mem} ]; then
            (( total_num_cpus += ${cpus} ))
            (( total_num_executors += ${executors_by_cpu} ))
        else
            (( total_num_cpus += ${executors_by_mem} * ${EXECUTOR_CORES} ))
            (( total_num_executors += ${executors_by_mem} ))
        fi
    done

    partitions=$(( ${total_num_cpus} * ${PARTITION_MULTIPLIER} ))
    cat >> ${DEFAULTS_FILE} << EOF
spark.executor.cores ${EXECUTOR_CORES}
spark.sql.shuffle.partitions ${partitions}
spark.executor.memory ${executor_memory_gb}g
EOF
    echo "Configured Spark to start ${total_num_executors} executors"
    echo "Set spark.sql.shuffle.partitions=${partitions} and" \
         "spark.executor.memory=${executor_memory_gb}g"
}

# Configure driver settings in spark-defaults.conf.
function config_driver()
{
    cat >> ${DEFAULTS_FILE} << EOF
spark.driver.memory ${DRIVER_MEMORY_GB}g
spark.driver.maxResultSize ${DRIVER_MEMORY_GB}g
EOF
    echo "Set driver memory to ${DRIVER_MEMORY_GB}g"
}

# Enable the history server in spark-defaults.conf.
function enable_history_server() {
    events_dir=${CONFIG_DIR}/events
    mkdir -p ${events_dir}
    cat >> ${DEFAULTS_FILE} << EOF
spark.eventLog.enabled true
spark.eventLog.compress true
spark.history.fs.cleaner.enabled true
spark.history.fs.cleaner.interval 1d
spark.history.fs.cleaner.maxAge 7d
spark.eventLog.dir file://${events_dir}
spark.history.fs.logDirectory file://${events_dir}
EOF
    echo "Enabled Spark history server at ${events_dir}"
}

# Enable all parameters related to dynamic allocation in spark-defaults.conf.
function enable_dynamic_allocation() {
    cat >> ${DEFAULTS_FILE} << EOF
spark.dynamicAllocation.enabled true
spark.dynamicAllocation.shuffleTracking.enabled true
spark.shuffle.service.enabled true
spark.shuffle.service.db.enabled = true
spark.worker.cleanup.enabled = true
EOF
    echo "Enabled dynamic allocation"
}

# Make a copy of spark-defaults.conf from the template. Keep a history of 10 files.
function copy_defaults_template_file() {
    grep -v "^\s*#\|^\s*$" ${DEFAULTS_TEMPLATE_FILE}
    ret=$?
    if [ $ret -eq 0 ]; then
        error "${DEFAULTS_TEMPLATE_FILE} cannot contain a line with uncommented settings"
    fi
    rotate_file=${CONFIG_DIR}/conf/rotate_spark_defaults.conf
    cat > ${rotate_file} << EOF
"${DEFAULTS_FILE}" {
    rotate 9
}
EOF
    if [ -f ${DEFAULTS_FILE} ]; then
        logrotate -f -s ${CONFIG_DIR}/conf/defaults.state ${rotate_file}
    fi
    cp ${DEFAULTS_TEMPLATE_FILE} ${DEFAULTS_FILE}
}

# Main

read -r -d "" USAGE << EOM
Usage: $(basename $0) [OPTION]... [SLURM_JOB_ID]...
Configure settings in spark-defaults.conf based on hardware resources in one or more SLURM job IDs.
Reads SLURM_JOB_ID from the environment.

Options:
  -D|--dynamic-allocation        Defaults to false.
  -H|--history-server            Defaults to false.
  -M|--driver-memory-gb VAL      Defaults to ${DRIVER_MEMORY_GB} (GB).
  -c|--executor-cores VAL        Defaults to ${EXECUTOR_CORES}.
  -d|--directory VAL             Defaults to current directory.
  -m|--partition-multiplier VAL  Set spark.sql.shuffle.partitions to number of
                                 cores multiplied by this value. Defaults to ${PARTITION_MULTIPLIER}.

Example:
  $(basename $0) --history-server --driver-memory-gb 2
EOM

while [[ $# -gt 0 ]]; do
  case $1 in
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
    -c|--executor-cores)
      EXECUTOR_CORES=${2}
      shift
      shift
      ;;
    -d|--directory)
      CONFIG_DIR=$(realpath ${2})
      shift
      shift
      ;;
    -h|--help)
      echo "${USAGE}"
      exit 0
      ;;
    -m|--shuffle-partitions-multiplier)
      PARTITION_MULTIPLIER=${2}
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

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

if ! [ -d ${CONFIG_DIR} ]; then
    error "CONFIG_DIR=${CONFIG_DIR} does not exist"
fi

num_jobs=${#SLURM_JOB_IDS[@]}
if [ ${num_jobs} -eq 0 ]; then
    if ! [ -z ${SLURM_JOB_ID} ]; then
        SLURM_JOB_IDS+=${SLURM_JOB_ID}
    else
        error "At least one SLURM job ID must be passed"
    fi
fi

DEFAULTS_FILE=${CONFIG_DIR}/conf/spark-defaults.conf
DEFAULTS_TEMPLATE_FILE=${DEFAULTS_FILE}.template
if ! [ -f ${DEFAULTS_TEMPLATE_FILE} ]; then
    error "${DEFAULTS_TEMPLATE_FILE} does not exist"
fi

module load singularity-container
check_errors
copy_defaults_template_file
config_executors
config_driver
if [ ${ENABLE_HISTORY_SERVER} = true ]; then
    enable_history_server
fi
if [ ${ENABLE_DYNAMIC_ALLOCATION} = true ]; then
    enable_dynamic_allocation
fi

echo "Configured settings in ${DEFAULTS_FILE}"
