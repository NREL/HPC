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

rm -f shutdown
srun collect_stats.sh . &

${SCRIPT_DIR}/start_spark_cluster.sh $SLURM_JOB_ID
# This runs an example script inside the container.
singularity run instance://spark spark-submit --master spark://$(hostname):7077 /opt/spark/examples/src/main/python/pi.py 500
${SCRIPT_DIR}/stop_spark_cluster.sh

touch shutdown
srun wait_for_stats.sh
rm shutdown
