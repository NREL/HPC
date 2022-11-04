export LUSTRE_BIND_MOUNTS="-B /lustre:/lustre \
    -B /nopt:/nopt \
    -B /projects:/projects \
    -B /scratch:/scratch"

function get_config_variable()
{
    name=$1
    if [ -z $name ]; then
        echo "Error: name must be provided"
        exit 1
    fi
    if [ -z ${CONFIG_DIR} ]; then
        echo "Error: CONFIG_DIR is not set"
        exit 1
    fi
    var=$(grep "^${name}\s*=" ${CONFIG_DIR}/config | awk -F "=" '{ print $2}' | tr -d " ")
    if [ -z ${var} ]; then
        echo "Error: Failed to parse config variable ${name}"
        exit 1
    fi
    echo "${var}"
}

export CONTAINER=$(get_config_variable "container")
export CONTAINER_INSTANCE_NAME=$(get_config_variable "container_instance_name")
export MASTER_NODE_MEMORY_OVERHEAD_GB=$(get_config_variable "master_node_memory_overhead_gb")
export WORKER_NODE_MEMORY_OVERHEAD_GB=$(get_config_variable "worker_node_memory_overhead_gb")

function get_memory_kb()
{
    echo "$(grep "MemTotal.*kB" /proc/meminfo | awk '{print $2}')"
}

function get_spark_bind_mounts()
{
    echo "-B ${CONFIG_DIR}/conf/:/opt/spark/conf"
}

function exec_spark_process()
{
    if [ -z ${1} ]; then
        echo "Error: A command must be passed to exec_spark_process"
        exit 1
    fi
    cmd=$@
    singularity exec \
        ${LUSTRE_BIND_MOUNTS} \
        $(get_spark_bind_mounts ${CONFIG_DIR}) \
        instance://${CONTAINER_INSTANCE_NAME} \
        ${cmd}
    ret=$?
    if [[ $ret -ne 0 ]]; then
        echo "Failed to exec Spark command=[${cmd}]: ${ret}"
        exit $ret
    fi
}
