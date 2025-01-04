#!/bin/bash

if [ -z ${1} ]; then
    echo "Error: CONFIG_DIR must be passed to start_container.sh"
    exit 1
fi
export CONFIG_DIR=$(realpath ${1})
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. ${SCRIPT_DIR}/common.sh

module load ${CONTAINER_MODULE}

echo "Start apptainer instance on $(hostname)"
apptainer instance start \
    -B $(mktemp -d ${CONFIG_DIR}/run/`hostname`_XXXX):/run \
    -B ${CONFIG_DIR}/dropbear/:/etc/dropbear \
    ${LUSTRE_BIND_MOUNTS} \
    $(get_spark_bind_mounts ${CONFIG_DIR}) \
    ${CONTAINER_PATH} \
    ${CONTAINER_NAME}
if [ $? -ne 0 ]; then
    exit 1
fi
apptainer exec instance://${CONTAINER_NAME} service dropbear start

enable_pg=$(get_config_variable "enable_postgres_metastore")
pg_password=$(get_config_variable "postgres_password")
if ${enable_pg}; then
    pg_data_dir=$(get_config_variable "postgres_data_dir")
    pg_run_dir=$(get_config_variable "postgres_run_dir")
    mkdir -p ${pg_data_dir} ${pg_run_dir}
    apptainer instance start \
        --env POSTGRES_PASSWORD=${pg_password} \
        ${LUSTRE_BIND_MOUNTS} \
        -B ${pg_data_dir}:/var/lib/postgresql/data \
        -B ${pg_run_dir}:/var/run/postgresql \
        docker://postgres \
        pg-server
fi
