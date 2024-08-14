SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

function error()
{
    echo "Error: ${@}"
    exit 1
}

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

export CONTAINER_PATH=$(get_config_variable "container")
export CONTAINER_NAME=$(get_config_variable "container_instance_name")
export NODE_MEMORY_OVERHEAD_GB=$(get_config_variable "node_memory_overhead_gb")

# Note: NREL_CLUSTER is always set on Kestrel nodes.
# It is not set on Eagle if you ssh into one of your allocated nodes, which is what these
# scripts do.
if [ ! -z ${NREL_CLUSTER} ] && [ ${NREL_CLUSTER} == "kestrel" ]; then
    module load apptainer
    export CONTAINER_MODULE=apptainer
    export CONTAINER_EXEC=apptainer
    export LUSTRE_BIND_MOUNTS=" -B /nopt:/nopt \
        -B /projects:/projects \
        -B /scratch:/scratch \
        -B /datasets:/datasets \
        -B /kfs2:/kfs2 \
        -B /kfs3:/kfs3"
else
    export CONTAINER_MODULE=singularity-container
    export CONTAINER_EXEC=singularity
    export LUSTRE_BIND_MOUNTS=" -B /nopt:/nopt \
        -B /datasets:/datasets \
        -B /lustre:/lustre \
        -B /projects:/projects \
        -B /scratch:/scratch"
fi

function get_spark_bind_mounts()
{
    echo "-B ${CONFIG_DIR}/conf/:/opt/spark/conf"
}

function get_spark_driver_memory_gb()
{
    cfile=${CONFIG_DIR}/conf/spark-defaults.conf
    mem=$(grep ^spark.driver.memory ${cfile} \
        | sed -E "s/spark.driver.memory\s*=*\s*([[:digit:]]+)g/\1/")
    echo "${mem}" | grep spark
    if [ $? -eq 0 ]; then
        echo "Did not find spark.driver.memory in ${cfile}"
        exit 1
    fi
    echo "${mem}"
}

function get_node_memory_overhead_gb()
{
    if is_heterogeneous_slurm_job; then
        echo "${NODE_MEMORY_OVERHEAD_GB}"
    else
        driver_mem=$(get_spark_driver_memory_gb)
        node_memory_overhead_gb=$(( ${driver_mem} + ${NODE_MEMORY_OVERHEAD_GB} ))
        echo "${node_memory_overhead_gb}"
    fi
}

function get_num_workers()
{
    master_node=$(hostname | tr -d '\n')
    num_workers=0
    for node_name in $(${SCRIPT_DIR}/get_node_names.sh ${SLURM_JOB_IDS[@]}); do
        if !(is_heterogeneous_slurm_job && [ "${node_name}" == "${master_node}" ] ); then
            (( num_workers += 1 ))
        fi
    done

    echo "${num_workers}"
}

function get_worker_memory_gb()
{
    # Spark documentation recommends only using 75% of system memory,
    # leaving the rest for the OS and buffer cache.
    # https://spark.apache.org/docs/latest/hardware-provisioning.html#memory
    if is_heterogeneous_slurm_job; then
        memory_gb=$(( ${SLURM_MEM_PER_NODE_HET_GROUP_1} / 1024 * 3 / 4 ))
    else
        memory_gb=$(( ${SLURM_MEM_PER_NODE} / 1024 * 3 / 4 ))
    fi
    echo "${memory_gb}"
}

function get_worker_num_cpus()
{
    if is_heterogeneous_slurm_job; then
        num_cpus=$(echo ${SLURM_JOB_CPUS_PER_NODE_HET_GROUP_1} | awk -F "(" '{ print $1}')
    else
        num_cpus=${SLURM_CPUS_ON_NODE}
    fi
    echo "${num_cpus}"
}

function is_heterogeneous_slurm_job()
{
    if [ -z ${SLURM_HET_SIZE} ]; then
        return 1
    fi
    return 0
}

function is_worker_node()
{
    if [ -z $1 ]; then
        echo "A node name must be passed to is_worker_node()"
        exit 1
    fi
    if [ -z ${SLURM_NODELIST_HET_GROUP_0} ]; then
        return 0
    fi
    node=$1
    if [ ${node} == ${SLURM_NODELIST_HET_GROUP_0} ]; then
        return 1
    fi
    return 0
}

function exec_spark_process()
{
    if [ -z ${1} ]; then
        echo "Error: A command must be passed to exec_spark_process"
        exit 1
    fi
    cmd=$@
    ${CONTAINER_EXEC} exec \
        ${LUSTRE_BIND_MOUNTS} \
        $(get_spark_bind_mounts ${CONFIG_DIR}) \
        instance://${CONTAINER_NAME} \
        ${cmd}
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "Failed to exec Spark command=[${cmd}]: ${ret}"
        exit $ret
    fi
}

function check_history_server_enabled()
{
    # $? will be 0 if the history server is enabled
    grep "^\s*spark\.eventLog\.enabled\s*=*\s*true" ${CONFIG_DIR}/conf/spark-defaults.conf
}

function run_checks()
{
    if [ -z ${SLURM_MEM_PER_NODE} ]; then
        echo "SLURM_MEM_PER_NODE is not set. Please submit the Slurm job with --mem, such as --mem=240000"
        exit 1
    fi
    if is_heterogeneous_slurm_job; then
        if [ ${SLURM_HET_SIZE} -gt 2 ]; then
            echo "A heterogeneous job can only have two groups: ${SLURM_HET_SIZE}"
            exit 1
        fi
        if [ ${SLURM_JOB_NUM_NODES_HET_GROUP_0} != "1" ]; then
            echo "SLURM_JOBID_HET_GROUP_0 can only have one node: ${SLURM_JOB_NUM_NODES_HET_GROUP_0}"
            exit 1
        fi
    fi
}
