# Running Apache Spark Clusters on an HPC

The scripts in this directory create ephemeral Apache Spark clusters on HPC compute nodes.

[Prerequisites](#prerequisites) | [Setup](#setup) | [Compute Nodes](#compute-nodes)

[Basic Configuration](#basic-configuration-instructions) | [Advanced Configuration](#advanced-configuration-instructions) | [Run Jobs](#run-jobs)

[Debugging Problems](#debugging-problems) | [Performance Monitoing](#performance-monitoring)

[Resources](#resources)

## Prerequisites
The scripts require the Spark software to be installed in an Apptainer container. The
`docker` directory includes Dockerfiles that build images derived from the base Apache
Spark Python and R images. The files have instructions on how to convert the Docker images to
Apptainer.

**Note**: If you are running on Eagle, replace `apptainer` with `singularity`. It was rebranded
and Eagle has the old software.

Kestrel:
```
$ module load apptainer
$ apptainer --help
```
Eagle:
```
$ module load singularity-container
$ singularity --help
```

Existing containers on Kestrel and Eagle:

- Spark 3.5.0 and Python 3.11, 3.12 (default = 3.12):
  - This image includes the packages `ipython`, `jupyter`, `numpy`, `pandas`, and `pyarrow`.
  - Kestrel: `/kfs2/pdatasets/images/apache_spark/spark350_py311.sif`
  - Eagle: `/datasets/images/apache_spark/spark350_py311.sif`

- Spark 3.3.1 and R 4.0.4:
  - This image includes the packages `tidyverse`, `sparklyr`, `data.table`, `here`, `janitor`, and
    `skimr`.
  - Kestrel: `/kfs2/pdatasets/images/apache_spark/spark_r.sif`
  - Eagle: `/datasets/images/apache_spark/spark_r.sif`

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

## Compute Nodes
Consider the type of compute nodes to acquire. If you will be performing large shuffles then
you must get nodes with fast local storage. A minimum requirement is approximately 500 MB/s.

- Any modern SSD should be sufficient.
- A spinning disk will be too slow.
- The Lustre filesystem on Kestrel may be fast enough for some queries.
- The Lustre filesystem on Eagle is too slow.
- A RAM drive (`/dev/shm`) will work well for smaller data sizes. Half the node memory is available
on NREL compute nodes.

### Kestrel
Standard nodes do not have local storage and are unsuitable if you will shuffle data. By default,
Spark will write to its tmp filesystem and that will quickly run out of space.

Kestrel has 256 nodes with local SSDs. Pick those if you can. For example, `salloc --tmp=1600G`.
Refer to the [Kestrel documentation](https://www.nrel.gov/hpc/kestrel-system-configuration.html) for
more information.

### Eagle
Standard nodes have spinning disks and are unsuitable if you will shuffle data. `bigmem` and `gpu`
nodes have fast local SSDs. Use those if you can. For example, `salloc --mem=730G` will acquire
nodes from either partition. Refer to the [Eagle
documentation](https://www.nrel.gov/hpc/eagle-system-configuration.html) for
more information.

## Basic Configuration Instructions
This section lists instructions that should be sufficient for most cases. Refer to [Advanced
Instructions](#advanced-instructions) to customize the Spark configuration parameters.

1. Acquire interactive compute nodes. Here are examples using the `debug` partition on Kestrel and
Eagle with ideal node types.

    Kestrel:
    ```
    $ salloc -t 01:00:00 -N2 --account=<your-account> --partition=debug --tmp=1600G
    ```
    
    Eagle:
    ```
    $ salloc -t 01:00:00 -N2 --account=<your-account> --partition=debug --mem=730G
    ```

2. Configure and start the cluster with `configure_and_start_spark.sh`. You can run
`configure_and_start_spark.sh --help` to see all options.

    This example assumes that you were automatically logged into the first compute node in your
    allocation by `salloc` and the environment variable `SLURM_JOB_ID` is set.
    
    Pass the path of the container that you want to use with the `-c` option.

    ```
    $ configure_and_start_spark.sh -c /kfs2/pdatasets/images/apache_spark/spark350_py311.sif
    ```

    **Note**: If you logged into the compute node manually with ssh or are using multiple
    Slurm allocations, specify the Slurm job IDs on the command line, like this:
    ```
    $ configure_and_start_spark.sh -c /kfs2/pdatasets/images/apache_spark/spark350_py311.sif <SLURM_JOB_ID_1> <SLURM_JOB_ID_2>
    ```

    **Note**: This command creates several directories that are used by the cluster. If you will
    run multiple clusters simultaneously, be sure to use different base directories. The base
    directory is the current directory by default, and can be changed with `--directory`.

3. Set this environment variable so that your jobs use the Spark configuration settings that you
just created.
    ```
    $ export SPARK_CONF_DIR=$(pwd)/conf
    ```

4. Run jobs as described [below](#run-jobs).

## Advanced Configuration Instructions
This section describes how to run the configuration scripts individually so that you can customize
any Spark parameter.

1. Acquire compute nodes as discussed above. The rest of the instructions assume that you are
logged into the first node in the allocation (either through `ssh` or automatically through
`salloc`).

2. Create a local copy of the configuration files with the command below. Specify an alternate
destination directory with `-d <directory>`.

    ```
    $ create_config.sh -c <path-to-spark-container>
    ```

3. Edit the `config` file if necessary. Note that the rest of this section relies on the setting
`container_instance_name = spark`.

4. Optional but recommended: Auto-configure the Spark parameters with `configure_spark.sh`. This
   will adjust the parameters based on the actual memory and CPU resources in your compute nodes.
   You can run `configure_spark.sh --help` to see all options.

    ```
    $ configure_spark.sh 
    ```

5. Optional: Customize the Spark configuration parameters.

This can be done by:

  - Editing the files in the `conf` directory.
  - Setting the parameters as CLI options when you run `spark-submit` or `pyspark`. Refer to the
  CLI help.

**Note**: The only way to customize Spark worker parameters is by editing `conf/spark-env.sh`.
You must make any changes before starting the cluster.

Here are some parameters in the `conf` files to consider editing:

**log4j2.properties**:
   - `rootLogger.level`: Spark is verbose when the log level is `info`. Change the level to
     `warn` if desired.

**spark-env.sh**:
   - `SPARK_LOG_DIR`: The Spark processes will log to this directory.
   - `SPARK_LOCAL_DIRS`: Spark will write temporary files here. It must be fast. Set it to
   `/dev/shm` if you want to use a RAM drive. Note that HPC nodes allow use of half of system
   memory. Adjust other parameters accordingly. If on Kestrel, you may be able to set it to a
   directory on the Lustre filesystem.
   - `SPARK_WORKER_DIR`: The Spark worker processes will log to this directory
     and use it for scratch space. It is configured to go to `/tmp/scratch` by default. Change it
     or copy the files before relinquishing the nodes if you want to preserve the files. They can
     be useful for debugging errors.

**spark-defaults.conf**:
   - `spark.executor.cores`: Online recommendations say that there is minimal parallelization
   benefit if the value is greater than 5. It should be configured in tandem with
   `spark.executor.memory` so that you maxmize the number of executors on each worker node.
   - `spark.executor.memory`: Adjust as necessary depending on the type of nodes you acquire. 10 GB
   and 5 cores per executor allow for 7 executors on Eagle and 20 on Kestrel, respectively, and
   will work well in most cases.
   - `spark.driver.memory`: Adjust as necessary depending on how much data you will pull from Spark
   into your application. Some online source recommend making it the same as
   `spark.executor.memory`.
   - `spark.eventLog.dir` and `spark.history.fs.logDirectory`: These directories must exist and
   will be used to store Spark history. If this is enabled, you can start a Spark history server
   after your jobs finish and review all jobs in the Spark UI. Disable these and
   `spark.eventLog.enabled` if you don't want to preserve the history.
   - `spark.sql.execution.arrow.pyspark.enabled`: Set it to `true` if you will use Python and
   convert Spark DataFrames to Pandas DataFrames.

The Spark documentation and other online resources provide additional information about tuning
these settings. Some resources are linked [below](#resources).

5. Start the Spark cluster.
    ```
    $ start_spark_cluster.sh
    ```

6. Set this environment variable so that your jobs use the Spark configuration settings that you
just created.
    ```
    $ export SPARK_CONF_DIR=$(pwd)/conf
    ```

7. Run jobs as described [below](#run-jobs).

## Run Jobs
These instructions assume that have allocated compute nodes and started a Spark cluster.

### Manual mode

1. Load the Apptainer environment if you want to run with its software. You can also run in your
own environment as long as you have the same versions of Spark and Python or R.

    ```
    $ module load apptainer
    ```

2. If you run in your own environment and want to use the configuration settings created by the
scripts, set the environment variable `SPARK_CONF_DIR`.

    ```
    $ export SPARK_CONF_DIR=$(pwd)/conf
    ```

3. Start a Spark process.

Refer to Python instructions [here](python.md).

Refer to R instructions [here](r.md).

### Batched execution
This directory includes sbatch script examples for each of the above execution types.

Refer to the scripts in the `slurm_scripts` directory.

### Mounts
The configuration scripts mount the following directories inside the container, and so you should
be able to load data files in any of them:
- `/projects`
- `/scratch`
- `/datasets`
- `/lustre` (Eagle)
- `/kfs2` (Kestrel)
- `/kfs3` (Kestrel)

### Shut down the cluster
This command will stop the Spark processes and the containers.
```
$ stop_spark_cluster.sh
```

## Debugging problems
Open the Spark web UI to observe what's happening with your jobs. You will have to forward ports
8080 and 4040 of the master node (first node in your Slurm allocation) through an ssh tunnel.

This is a Mac/Linux example to create a tunnel. On Windows adjust the environment variable syntax as
needed for the Command shell or PowerShell.
```
$ export COMPUTE_NODE=<your-compute-node-name>
$ ssh -L 4040:$COMPUTE_NODE:4040 -L 8080:$COMPUTE_NODE:8080 $USER@kestrel.hpc.nrel.gov

Open your browser to http://localhost:4040 after configuring the tunnel to access the application UI.

Before inspecting job details you may first want to confirm that the correct Spark configuration
settings are in effect by looking at the `Environment` tab.

If you enable the history server then you can open this UI after you relinquish the nodes. Here
is an example of how to start it:

```
$ apptainer exec \
	--env SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=./events" \
	instance://spark \
	start-history-server.sh
```

**Note**: Be sure to gracefully shut down the cluster with `stop_spark_cluster.sh` if you intend
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
$ ./ssh-multi $(./scripts/get-node-names $SLURM_JOB_ID)
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
