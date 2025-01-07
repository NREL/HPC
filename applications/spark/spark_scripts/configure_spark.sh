#!/bin/bash

CONFIG_DIR=$(pwd)
DRIVER_MEMORY_GB=10
ENABLE_DYNAMIC_ALLOCATION=false
ENABLE_HISTORY_SERVER=false
# Many online docs say executors max out with 5 threads.
EXECUTOR_CORES=5
PARTITION_MULTIPLIER=1
SLURM_JOB_IDS=()
SPARK_SCRATCH="tmpfs"

# Configure executor settings in spark-defaults.conf.
function config_executors()
{
    num_workers=$(get_num_workers)
    node_memory_overhead_gb=$(get_node_memory_overhead_gb)
    worker_memory_gb=$(get_worker_memory_gb)
    worker_num_cpus=$(get_worker_num_cpus)

    # Leave one CPU for OS and management software.
    worker_num_cpus=$(( ${worker_num_cpus} - 1 ))

    min_executors_per_node=$(( ${worker_num_cpus} / ${EXECUTOR_CORES} ))
    executor_memory_gb=$(( ${worker_memory_gb} / ${min_executors_per_node} ))
    executors_by_mem=$(( ${worker_memory_gb} / ${executor_memory_gb} ))
    executors_by_cpu=$(( ${worker_num_cpus} / ${EXECUTOR_CORES} ))
    if [ ${executors_by_cpu} -le ${executors_by_mem} ]; then
        executors_per_node=${executors_by_cpu}
    else
        executors_per_node=${executors_by_mem}
    fi

    total_num_cpus=$(( ${executors_per_node} * ${EXECUTOR_CORES} * ${num_workers} ))
    total_num_executors=$(( ${executors_per_node} * ${num_workers} ))
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
Usage: $(basename $0) [OPTIONS]... [SLURM_JOB_ID]...

  Configure settings in spark-defaults.conf based on hardware resources in one or more SLURM job IDs.
  Reads SLURM_JOB_ID from the environment.

Options:
  -D, --dynamic-allocation            Enable dynamic resource allocation. [Default: false]
  -H, --history-server                Enable the history server. [Default: false]
  -M, --driver-memory-gb INTEGER      Driver memory in GB. [Default: ${DRIVER_MEMORY_GB}]
  -e, --executor-cores INTEGER        Number of cores per executor. [Default: ${EXECUTOR_CORES}]
  -d, --directory TEXT                Base directory with configuration files. [Default: current]
  -l, --spark-scratch TEXT            Directory given to Spark workers for shuffle writes and log files.
                                      [Default: compute node tmpfs]
  -m, --partition-multiplier INTEGER  Set spark.sql.shuffle.partitions to number of
                                      cores multiplied by this value. [Default: ${PARTITION_MULTIPLIER}]

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
    -e|--executor-cores)
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

function write_worker_nodes()
{
    workers_file="${CONFIG_DIR}/conf/workers"
    rm -f ${workers_file}
    touch ${workers_file}

    for node in $(${SCRIPT_DIR}/get_node_names.sh ${SLURM_JOB_IDS[@]})
    do
        if is_worker_node ${node}; then
            echo "${node}" >> $workers_file
        fi
    done
}

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

run_checks

DEFAULTS_FILE=${CONFIG_DIR}/conf/spark-defaults.conf
DEFAULTS_TEMPLATE_FILE=${DEFAULTS_FILE}.template
if ! [ -f ${DEFAULTS_TEMPLATE_FILE} ]; then
    error "${DEFAULTS_TEMPLATE_FILE} does not exist"
fi

module load ${CONTAINER_MODULE}
copy_defaults_template_file
enable_derby_metastore=$(get_config_variable "enable_derby_metastore")
enable_postgres_metastore=$(get_config_variable "enable_postgres_metastore")
metastore_dir=$(get_config_variable "metastore_dir")
if ${enable_derby_metastore} || ${enable_postgres_metastore}; then
    if ${enable_postgres_metastore}; then
        write_postgres_hive_site_file ${CONFIG_DIR}/conf
    elif ${enable_derby_metastore}; then
        cp ${CONFIG_DIR}/conf/hive-site.xml.template ${CONFIG_DIR}/conf/hive-site.xml
        sed -i "s|REPLACE_ME_WITH_CUSTOM_PATH|${metastore_dir}/metastore_db|" ${CONFIG_DIR}/conf/hive-site.xml
    fi
    echo "spark.sql.warehouse.dir ${metastore_dir}/spark-warehouse" >> ${CONFIG_DIR}/conf/spark-defaults.conf
else
    rm -f ${CONFIG_DIR}/conf/hive-site.xml
fi

config_driver
write_worker_nodes
config_executors
if ${ENABLE_HISTORY_SERVER}; then
    enable_history_server
fi
if ${ENABLE_DYNAMIC_ALLOCATION}; then
    enable_dynamic_allocation
else
    enable_pg=$(get_config_variable "enable_postgres_metastore")
    enable_derby=$(get_config_variable "enable_derby_metastore")
    if ${enable_pg} || ${enable_derby}; then
        echo "Enable dynamic resource allocation because a hive metastore is enabled."
        enable_dynamic_allocation
        echo "spark.driver.extraClassPath /datasets/images/apache_spark/postgresql-42.7.4.jar" >> ${DEFAULTS_FILE}
        echo "spark.executor.extraClassPath /datasets/images/apache_spark/postgresql-42.7.4.jar" >> ${DEFAULTS_FILE}
    fi
fi
if [ "${SPARK_SCRATCH}" != "tmpfs" ]; then
    spark_scratch=$(realpath ${SPARK_SCRATCH})
    echo "SPARK_LOCAL_DIRS=${spark_scratch}/local" >> ${CONFIG_DIR}/conf/spark-env.sh
    echo "SPARK_WORKER_DIR=${spark_scratch}/worker" >> ${CONFIG_DIR}/conf/spark-env.sh
    echo "Configured Spark workers to use ${spark_scratch} for shuffle data and log files."
fi

echo "Configured settings in ${DEFAULTS_FILE}"
