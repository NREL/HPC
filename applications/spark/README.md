# How to run Apache Spark on Kestrel

This page demonstrates how to create ephemeral Apache Spark clusters on HPC
compute nodes using the Python package [sparkctl](https://github.com/NREL/sparkctl).

Please refer to the [sparkctl documentation](https://nrel.github.io/sparkctl/) for tutorials
and how-tos. This page gives a few short examples to illustrate what is possible.

## Installation
1. Create a Python virtual environment. This example uses `~/python-envs`, but you can use any
   location.

   ```console
   $ module load python
   $ mkdir ~/python-envs
   $ python -m venv ~/python-envs/sparkctl
   ```

   Activate the environment and install `sparkctl`. Run `deactivate` whenever you are finished.
   ```console
   $ source ~/python-envs/sparkctl/bin/activate
   $ pip install "sparkctl[pyspark]"
   ```

2. Create a global configuration file pointing to the Spark software. The paths below are valid
   for Kestrel.

   ```
   $ sparkctl default-config \
       /datasets/images/apache_spark/spark-4.0.0-bin-hadoop3 \
       /datasets/images/apache_spark/jdk-21.0.7 \
       --hadoop-path /datasets/images/apache_spark/hadoop-3.4.1 \
       --hive-tarball /datasets/images/apache_spark/apache-hive-4.0.1-bin.tar.gz \
       --postgresql-jar-file /datasets/images/apache_spark/jdk-21.0.7 \
       --compute-environment slurm
   ```

## Usage
1. Allocate compute node(s). For demonstration purposes, a debug node is sufficient. For real work
   we recommend a hetergeneous allocation with a partial shared node for the Spark master and complete
   nodes for Spark workers.

   Refer to this [page](https://nrel.github.io/sparkctl/reference/hpc/kestrel.html#compute-nodes)
   for specific guidance on Kestrel compute nodes.

   Debug node:
   ```console
   $ salloc --account=<your-account> -N1 -t 1:00:00 --mem=240G --partition=debug
   ```

   Hetergeneous nodes:
   ```console
   $ salloc --account=<your-account> -t 01:00:00 -n4 --mem=30G --partition=shared : -N4 --mem=240G
   ```

2. Run Spark jobs. `sparkctl` provides two general workflows.

   In the first workflow, you configure and start a cluster with the `sparkctl` CLI and then run jobs
   with `spark-submit`, `spark-shell`, `pyspark`, or some other Spark client from the Spark
   [downloads page](https://spark.apache.org/downloads.html).

   ```console
   $ sparkctl configure
   $ sparkctl start
   ```

   Follow instructions on the screen to export Spark/Java environment variables.
   ```console
   $ export SPARK_CONF_DIR=$(pwd)/conf
   $ export JAVA_HOME=/datasets/images/apache_spark/jdk-21.0.7
   ```

   As an example, copy/paste this code into `job.py`:
   ```python
   from pyspark.sql import SparkSession
   spark = SparkSession.builder.appName("my_app").getOrCreate()
   df = spark.createDataFrame([(x, x + 1) for x in range(100)], ["a", "b"])
   df.show()
   ```

   Run the job.
   ```console
   $ spark-submit --master spark://$(hostname):7077 job.py
   ```

   Stop the Spark cluster when complete.
   ```console
   $ sparkctl stop
   ```

   In the second workflow, you run Spark jobs in a Python script using the sparkctl library to manage
   the cluster. This workflow hides the details of starting/stopping the cluster and setting
   the Spark environment variables.

   ```python
   from sparkctl import ClusterManager, make_default_spark_config

   config = make_default_spark_config()
   mgr = ClusterManager(config)
   with mgr.managed_cluster() as spark:
       df = spark.createDataFrame([(x, x + 1) for x in range(100)], ["a", "b"])
       df.show()
   ```

Please refer to the Apache Spark [website](https://spark.apache.org/docs/latest/#where-to-go-from-here)
for more examples.
