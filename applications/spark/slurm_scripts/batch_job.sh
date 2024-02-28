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
# This runs an example script inside the container.
apptainer run instance://spark spark-submit --master spark://$(hostname):7077 /opt/spark/examples/src/main/python/pi.py 500
${SCRIPT_DIR}/stop_spark_cluster.sh
