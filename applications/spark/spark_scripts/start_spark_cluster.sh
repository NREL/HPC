#!/bin/bash

function setup()
{
    if ! [ -d dropbear ]; then
        apptainer exec ${LUSTRE_BIND_MOUNTS} ${CONTAINER_PATH} ${SCRIPT_DIR}/make_dropbear.sh
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
            if [ $? -ne 0 ]; then
                echo "Failed to start the container on ${node_name}"
                exit 1
            fi
        fi
    done

    echo "Started containers on all nodes"
}

function start_spark_processes()
{
    master_node=$(hostname | tr -d '\n')
    spark_cluster=spark://${master_node}:7077
    node_memory_overhead_gb=$(get_node_memory_overhead_gb)
    enable_pg=$(get_config_variable "enable_postgres_metastore")

    if ${enable_pg}; then
        setup_postgres_metastore
    fi

    exec_spark_process start-master.sh
    check_history_server_enabled
    if [ $? -eq 0 ]; then
        exec_spark_process start-history-server.sh
    fi
    enable_thrift_server=$(get_config_variable "thrift_server")
    if [ ${enable_thrift_server} == "true" ]; then
        exec_spark_process start-thriftserver.sh --master ${spark_cluster}
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

function setup_postgres_metastore()
{
    pg_password=$(get_config_variable "POSTGRES_PASSWORD")
    pg_data_dir=$(get_config_variable "postgres_data_dir")
    pg_run_dir=$(get_config_variable "postgres_run_dir")
    if [ -z "$(ls -A ${pg_data_dir})" ]; then
        pg_exists=false
    else
        pg_exists=true
    fi
    if ! ${pg_exists}; then
        apptainer exec instance://pg-server initdb
    fi
    set -e
    apptainer exec instance://pg-server \
        pg_ctl \
            -D /var/lib/postgresql/data \
            -l logfile \
            start
    if ! ${pg_exists}; then
        apptainer exec instance://pg-server createdb hive_metastore
        apptainer exec instance://pg-server \
            psql \
                -c "CREATE ROLE postgres WITH LOGIN SUPERUSER PASSWORD '${pg_password}'" \
                hive_metastore
        init_hive
    fi
    set +e
}

function init_hive()
{
    export HADOOP_HOME=/datasets/images/apache_spark/hadoop-3.4.1
    export HIVE_HOME=${CONFIG_DIR}/apache-hive-4.0.1-bin
    export HIVE_CONF=${CONFIG_DIR}/apache-hive-4.0.1-bin/conf
    export JAVA_HOME=/usr
    existing_dir=${CONFIG_DIR}/apache-hive-4.0.1-bin
    if [ -d ${existing_dir} ]; then
        rm -rf ${existing_dir}
    fi
    tar -C ${CONFIG_DIR} -xzf /datasets/images/apache_spark/apache-hive-4.0.1-bin.tar.gz
    cp /datasets/images/apache_spark/postgresql-42.7.4.jar ${HIVE_HOME}/lib
    write_postgres_hive_site_file ${HIVE_CONF}
    cd ${HIVE_CONF}
    ${HIVE_HOME}/bin/schematool -dbType postgres -initSchema
    cd -
    unset HADOOP_HOME HIVE_HOME HIVE_CONF JAVA_HOME
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
