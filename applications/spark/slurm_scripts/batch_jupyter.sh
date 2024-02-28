#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=spark_job
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.o
#SBATCH --error=output_%j.e
#SBATCH --nodes=2
#SBATCH --tmp=1600G
#SBATCH --partition=debug

module load apptainer
SCRIPT_DIR=~/repos/HPC/applications/spark/spark_scripts
${SCRIPT_DIR}/configure_and_start_spark.sh
apptainer run \
	--bind /lustre:/lustre \
	--bind /projects:/projects \
	--bind /scratch:/scratch \
	--bind /nopt:/nopt \
	--env PYSPARK_DRIVER_PYTHON=jupyter \
	--env PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=8889 --ip=0.0.0.0" \
	--net \
	--network-args \
	"portmap=8889:8889" \
	instance://spark \
    pyspark --master spark://$(hostname):7077

${SCRIPT_DIR}/stop_spark_cluster.sh
