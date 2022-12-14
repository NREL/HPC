#!/bin/bash
#SBATCH --account=<your-account>
#SBATCH --job-name=spark_job
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.o
#SBATCH --error=output_%j.e
#SBATCH --nodes=2
#SBATCH --partition=debug

module load singularity-container
SCRIPT_DIR=~/repos/HPC/applications/spark/spark_scripts
${SCRIPT_DIR}/configure_spark.sh
${SCRIPT_DIR}/start_spark_cluster.sh
# This runs an example script inside the container.
singularity run instance://spark spark-submit --code-examples spark://$(hostname):7077 /opt/spark/examples/src/main/python/pi.py 500
${SCRIPT_DIR}/stop_spark_cluster.sh
