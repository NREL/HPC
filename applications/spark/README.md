# Running Apache Spark Clusters on an HPC

The scripts in this directory create ephemeral Apache Spark clusters on HPC compute nodes.

## Prerequisites
The scripts require the Spark software to be installed in a Singularity container. The
`docker/python` directory includes a `Dockerfile` that builds an image derived from the base Apache
Spark Python image. (Apache has other images for Scala and R.) The file has instructions on how to
convert the Docker image to Singularity.

A Singularity image with Spark 3.3 and Python 3.9 is stored on Eagle at
`/scratch/dthom/containers/spark.sif`. This image includes `jupyter` to empower workflows in
notebooks.

## Setup
1. Copy the `config` file, `scripts` directory, and the `conf` directory to a location on the
   HPC.
2. Edit the `config` file as necessary. You may only need to modify the path to the Spark
   container.
3. Consider what type of compute nodes to acquire. If you will be performing shuffle operations
   then you must get nodes with fast local storage. `bigmem` and `gpu` nodes have local SSDs that
   can read/write at 2 GB/s. The standard nodes have spinning disks that can only read/write at
   ~130 MB/s. Your jobs will fail if you use those nodes. You can consider specifying a RAM disk
   as Spark local storage (`/dev/shm`), but you must be sure you have enough space.
4. Edit the files in `conf` as desired. These control the global Spark settings. You can
   also customize some of the `spark-defaults` parameters when you run `spark-submit` or `pyspark`.
   Refer to the CLI help.

   Here are some parameters to consider editing:

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
files, directories, and scripts.

The start script takes one or more SLURM job IDs as inputs. The script will detect the nodes and
start the container on each.

### Manual mode
1. Allocate nodes however you'd like (`salloc`, `sbatch`, `srun`).
2. Login to the first node if not already there.
3. Start the Spark cluster
If you allocated the nodes with `salloc`:
```
$ ./scripts/start-spark-cluster . $SLURM_JOB_ID
```
If you allocated two jobs separately and ssh'd into a node:
```
$ ./scripts/start-spark-cluster . <SLURM_JOB_ID1> <SLURM_JOB_ID2>
```

4. Load the Singularity container if you want to run with its software. You can also run in your
   own environment as long as you have the same versions of Spark and Python.
```
$ module load singularity-container
```
5. Start a Spark process.

#### Interactive Python interpreter
This uses ipython, which is optional.
```
$ singularity run --env PYSPARK_DRIVER_PYTHON=ipython instance://spark pyspark --master spark://`hostname`:7077
```
The Spark session object is available globally in the variable `spark`. Create or load dataframes with it.

#### Jupyter notebook
```
$ singularity run \
	--net \
	--network-args "portmap=8889:8889" \
	--env PYSPARK_DRIVER_PYTHON=jupyter \
	--env PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=8889 --ip=0.0.0.0" \
	instance://spark \
	pyspark --master spark://`hostname`:7077
```
The Jupyter process will print a URL to the terminal. You can access it from your laptop after you
forward the ports through an ssh tunnel.

This is a Mac/Linux example. On Windows adjust the environment variable syntax as needed for the Command shell
or PowerShell.
```
$ export COMPUTE_NODE=<your-compute-node-name>
$ ssh -L 4040:$COMPUTE_NODE:4040 -L 8080:$COMPUTE_NODE:8080 -L 8889:$COMPUTE_NODE:8889  $USER@eagle.hpc.nrel.gov
```
Open the link in your browser and start a new notebook. Connect to the Spark session by entering
this text into a cell.
```
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("my_session").getOrCreate()
```


#### Run a script
```
$ singularity run \
	instance://spark \
	spark-submit --master spark://`hostname`:7077 <your-script>
```
Note: if your script is Python, the filename must end in .py.


### Batched execution
This directory includes sbatch script examples for each of the above execution types.

Refer to the scripts in `slurm_scripts`.

## Python performance considerations
Refer to this Spark documentation if you will convert Spark DataFrames to Pandas DataFrames.
https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html

## Debugging problems
Open the Spark web UI to observe what's happening with your jobs. You will have to forward ports
8080 and 4040 of the master node (first node in your SLURM allocation) through an ssh tunnel.

Open your browser to http://localhost:4040 after configuring the tunnel.

If you enable the history server then you can open this UI after you relinquish the nodes. Here
is an example of how to start it:

```
$ singularity exec \
	--env SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=./events" \
	instance://spark \
	start-history-server.sh
```

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
