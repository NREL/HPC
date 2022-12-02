# Running Apache Spark Clusters on an HPC

The scripts in this directory create ephemeral Apache Spark clusters on HPC compute nodes.

## Prerequisites
The scripts require the Spark software to be installed in a Singularity container. The
`docker` directory includes Dockerfiles that build images derived from the base Apache
Spark Python and R images. The files have instructions on how to convert the Docker images to
Singularity.

Existing Singularity containers on Eagle:
- Spark 3.3.1 and Python 3.9 is at `/datasets/images/apache_spark/spark_py39.sif`.
This image includes the packages `ipython`, `jupyter`, `numpy`, `pandas`, and `pyarrow`.
- Spark 3.3.1 and $ 4.0.4 is at `/datasets/images/apache_spark/spark_r.sif`.
This image includes the packages `tidyverse`, `sparklyr`, `data.table`, `here`, `janitor`, and
`skimr`.

## Setup

1. Clone the repository:
```
$ git clone https://github.com/NREL/HPC.git
```
2. Change to a directory in `/scratch/$USER`.
3. Add the `spark_scripts` directory to your path.
```
$ export PATH=$PATH:<your-repo-path>/HPC/applications/spark/spark_scripts
```
4. Copy the `config` file and `conf` directory with the command below. Specify an alternate
destination directory with `-d <directory>`.
```
$ create_config.sh -c <path-to-spark-container>
```

5. Edit the `config` file if necessary. Note that the rest of this page relies on the setting
   `container_instance_name = spark`.
6. Consider what type of compute nodes to acquire. If you will be performing large shuffles
   then you must get nodes with fast local storage. `bigmem` and `gpu` nodes have local SSDs that
   can read/write at 2 GB/s. The standard nodes have spinning disks that can only read/write at
   ~130 MB/s. Your jobs will fail if you use those nodes. You can consider specifying a RAM disk
   as Spark local storage (`/dev/shm`), but you must be sure you have enough space.
7. Decide how and when you want to configure your Spark application parameters.

   - Manually specify global settings in `conf`. Note that worker settings in `conf/spark-env.sh`
   must be set before starting the cluster.
   - Auto-configure global settings with `configure_spark.sh`. You can run this script after
   acquiring compute nodes and it will apply settings based on the hardware resources (memory/CPU)
   of those nodes. Run `configure_spark.sh --help` to see available options.
   - At runtime when you run `spark-submit` or `pyspark`. Refer to the CLI help.
   
   Here are some parameters in the `conf` files to consider editing:

**log4j2.properties**:
   - `rootLogger.level`: Spark is verbose when the log level is `info`. Change the level to
     `warn` if desired.

**spark-env.sh**:
   - `SPARK_LOG_DIR`: The Spark processes will log to this directory.
   - `SPARK_LOCAL_DIRS`: Spark will write temporary files here. It must be fast. Set it to `/dev/shm`
     if you want to use a RAM disk. Note that Eagle nodes allow use of half of system memory. Adjust
     other parameters accordingly.
   - `SPARK_WORKER_DIR`: The Spark worker processes will log to this directory
     and use it for scratch space. It is configured to go to `/tmp/scratch` by default. Change it
     or copy the files before relinquishing the nodes if you want to preserve the files. They can
     be useful for debugging errors.

**spark-defaults.conf**:
   - `spark.executor.cores`: Online recommendations say that there is minimal parallelization benefit
     if the value is greater than 5. It should be configured in tandem with `spark.executor.memory`
     so that you maxmize the number of executors on each worker node. 7 executors work well on Eagle
     nodes (35 out of 36 available cores).
   - `spark.executor.memory`: Adjust as necessary depending on the type of nodes you acquire. Make
     it big enough for 7 executors after adjusting for overhead for OS and management processes.
   - `spark.driver.memory`: Adjust as necessary depending on how much data you will pull from Spark
     into your application.
   - `spark.eventLog.dir` and `spark.history.fs.logDirectory`: These directories must exist and
     will be used to store Spark history. If this is enabled, you can start a Spark history server
     after your jobs finish and review all jobs in the Spark UI. Disable these and
     `spark.eventLog.enabled` if you don't want to preserve the history.
   - `spark.sql.execution.arrow.pyspark.enabled`: Set it to `true` if you will use Python and
     convert Spark DataFrames to Pandas DataFrames.


## Usage
These instructions assume that you are running in a directory that contains the configuration
files and directories, and that you've added the `spark_scripts` directory to your `PATH`
environment variable.

The start script takes one or more SLURM job IDs as inputs. The script will detect the nodes and
start the container on each.

### Manual mode

**Note**: The best way to test this functionality is with an interactive session on bigmem nodes
in the debug partition.

Example command (2 nodes):
```
$ salloc -t 01:00:00 -N2 --account=<your-account> --partition=debug --mem=730G
```

1. Allocate nodes however you'd like (`salloc`, `sbatch`, `srun`).
2. Login to the first node if not already there.
3. Optional: Run `configure_spark.sh` to apply settings based on actual compute node resources.
4. Start the Spark cluster
If you allocated the nodes with `salloc`:
```
$ start_spark_cluster.sh
```
If you allocated two jobs separately and ssh'd into a node:
```
$ start_spark_cluster.sh <SLURM_JOB_ID1> <SLURM_JOB_ID2>
```

5. Load the Singularity environment if you want to run with its software. You can also run in your
   own environment as long as you have the same versions of Spark and Python or R.
```
$ module load singularity-container
```

6. If you run in your own environment, set the environment variable `SPARK_CONF_DIR` if
you want to use the configuration settings created by the scripts.

```
$ export SPARK_CONF_DIR=$(pwd)/conf
```

7. Start a Spark process.

Refer to Python instructions [here](python.md).

Refer to R instructions [here](r.md).

### Batched execution
This directory includes sbatch script examples for each of the above execution types.

Refer to the scripts in the `slurm_scripts` directory.

### Mounts
The configuration scripts mount the following directories inside the container, and so you should
be able to load data files in any of them:
- `/lustre`
- `/projects`
- `/scratch`
- `/datasets`


## Debugging problems
Open the Spark web UI to observe what's happening with your jobs. You will have to forward ports
8080 and 4040 of the master node (first node in your SLURM allocation) through an ssh tunnel.

Open your browser to http://localhost:4040 after configuring the tunnel to access the application UI.

Before inspecting job details you may first want to confirm that the correct Spark configuration
settings are in effect by looking at the `Environment` tab.

If you enable the history server then you can open this UI after you relinquish the nodes. Here
is an example of how to start it:

```
$ singularity exec \
	--env SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=./events" \
	instance://spark \
	start-history-server.sh
```

**Note**: Be sure to cleanly shutdown the cluster with `stop_spark_cluster.sh` if you intend
to look at the history.

## Performance monitoring
Tuning a Spark application and cluster can be difficult. It is advisable to monitor hardware
resource utilization on your compute nodes. You need to ensure that you are using all available
CPUs and are not bottle-necked by the storage.

Here is an example of how to run `htop` on multiple nodes simulataneously with `tmux`.

Download this script: https://raw.githubusercontent.com/johnko/ssh-multi/master/bin/ssh-multi

Run it like this:
```
$ ./ssh-multi node1 node2 nodeN
```
It will start `tmux` with one pane for each node and synchonize mode enabled. Typing in one pane
types everywhere.
```
$ htop
```
Ensure that you are using all CPUs.

You can monitor memory usage with
```
$ watch free -g
```

You can monitor disk usage with
```
$ iostat -t 1 -xm
```

Note that this directory includes a helper script to convert SLURM job IDs to node names.
Here is a shortcut for the above if you are on a node acquired with `salloc`:
```
$ ./ssh-multi `./scripts/get-node-names $SLURM_JOB_ID`
```

### Automated performance monitoring
You may want to run performance monitoring for the duration of your job and inspect the results
later. This section describes one way to do that with a separate tool.

1. Configure a Python virtual environment.
2. Install `jade`.
```
$ pip install NREL-jade
```
This package includes a tool that collects resource utilization data. You can run it like this:
```
$ jade stats collect --interval=1 --output=my-stats
```
The tool generates a Parquet file for each resource type as well as HTML plots.

3. Configure your `sbatch` script to run this tool on each node. Refer to the scripts in
   `slurm_scripts_with_resource_monitoring`. The output directories will contain HTML plots for
   each comupte node.


## Resources
- Spark cluster overview: https://spark.apache.org/docs/latest/cluster-overview.html
- Spark Python APIs: https://spark.apache.org/docs/latest/api/python/getting_started/index.html
- Spark tuning guide from Apache: https://spark.apache.org/docs/latest/tuning.html
- Spark tuning guide from Amazon: https://aws.amazon.com/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/
- Performance recommendations: https://www.youtube.com/watch?v=daXEp4HmS-E&t=4251s
