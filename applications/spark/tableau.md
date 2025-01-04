# Visualize Data with Tableau
[Tableau](https://www.tableau.com/) is a commercial tool for exploring and visualizing
tabular data. Licenses are available to NRELians.

In addition to making visualizations, Tableau makes it easy to select, filter, group, and describe
your data in tables. This can be easier than the same operations in a Python/R/SQL REPL with Spark.

If you have Tableau installed on your laptop, you can connect it to your Spark cluster on Kestrel.

In addition to Tableau, you must install this
[ODBC driver](https://www.databricks.com/spark/odbc-drivers-download) on your laptop. It may
require admin priviledges.

## Concepts
Configuring this workflow has some complexity, and so it is important that you understand what's
going on.

### Components
- Thrift Server: Connects SQL clients (like Tableau) to the Spark cluster over the network.
- Hive Metastore: Created by Spark in your working directory on the Lustre filesystem.
- Derby database: Manages the metastore by default. This has an important limitation: there can
  only be one open Spark session at a time. This means that you cannot simultaneously run jobs
  through PySpark and the Thrift Server.
- PostgreSQL database: Optional database for the metastore that supports multiple Spark sessions.
  This can be enabled in the script `configure_and_start_spark.sh`. It will start a postgres
  database and initialize it through Hive.
- Data sources: Parquet/CSV files on the Lustre filesystem.
- Views: Read-only views into the data sources that get registered in the database.

### Workflow
1. Start a Spark cluster.
2. Create one or more views into your data sources. This will create the metastore in the current
directory by default.
3. Start the Thrift Server.
4. Connect with a SQL client and send queries.

**Warning**: You can cause conflicts in the metastore if you try to run Spark commands from the
compute node while the Thrift Server is running. Only one process can access the database at a
time.

## Compute Node Instructions
1. Acquire one or more compute nodes. The rest of this section assumes that you are logged
into the head node of the Slurm allocation.

2. Start the Spark cluster.
   ```
   $ configure_and_start_spark.sh
   ```

3. Load the Apptainer environment so that you can run Spark commands inside the container.
   ```
   $ module load apptainer
   ```

4. Create one or more views. These examples show how to create one view on the command line and
how to create several views in a batch process.

   Suppose that your data sources are located at `/projects/my-project/data`.
   
   Create one view. (Note the backslashes that prevent the shell from acting on the backticks.)
   ```
   $ apptainer exec instance://spark spark-sql -e \
       "CREATE VIEW my_view AS SELECT * FROM parquet.\`/projects/my-project/data/table1.parquet\`"
   ```
   
   Create several views.
   
   Add lines like these to a text file called `views.txt`:
   ```
   CREATE VIEW my_view1 AS SELECT * FROM parquet.`/projects/my-project/data/table1.parquet;`
   CREATE VIEW my_view2 AS SELECT * FROM parquet.`/projects/my-project/data/table2.parquet;`
   CREATE VIEW my_view3 AS SELECT * FROM parquet.`/projects/my-project/data/table3.parquet;`
   ```
   
   Execute the commands with `spark-sql`.
   ```
   $ apptainer exec instance://spark spark-sql -f views.txt
   ```

5. Start the Thrift Server.
   ```
   $ apptainer exec instance://spark start-thriftserver.sh --master=spark://$(hostname):7077
   ```

   **Warning**: If you need to create more views, you must stop this server to avoid database
   conflicts (unless you enabled the Postgres option).
   ```
   $ apptainer exec instance://spark stop-thriftserver.sh
   ```

## Client-side Instructions

1. Create an ssh tunnel from your laptop to the compute node running the Thrift Server (the
head node in the Slurm allocation).

   Replace the text below with your compute node and username.
   ```
   $ ssh -L 10000:x3000c0s25b0n0:10000 jdoe@kestrel.hpc.nrel.gov
   ```

2. Open Tableau and connect to a new data source. Choose the `Spark SQL` connector installed
above. It may be listed in the `Connect` window under `To a Server` after selecting `More...`.

3. A `Spark SQL` window will open. Choose these settings:

   - Connection = `SparkThriftServer (Spark 1.1 and later)`
   - Server = `localhost`
   - Port = `10000`
   - Authentication = `Username`
   - Password: leave blank

4. Click `Sign In`. A Tableau workbook will open with a connection to `localhost`.

5. Select the `Schema` dropdown, type `default` and click the plus button.

6. A Table section will open. Click the search button next to `Enter table name`. Tableau should
discover the tables in your database and list them. Double-click on one.

## Persistent metastore and spark-warehouse
The default locations of the metastore and spark-warehouse are in your current directory.
You will notice the directories `metastore_db` and `spark-warehouse`.

If you have a set of static tables that you use frequently, you may want to create a persistent
metastore and spark-warehouse. You can set these locations with a custom option.

```
$ configure_and_start_spark.sh -s <your-path>
```
You may notice that this updates the files `conf/hive-site.xml` and `conf/spark-defaults.conf`.

All tables and views that you create will be available to any cluster started this way.

If you will only be accessing the tables and views through Tableau, you can start the Thrift Server
automatically.

```
$ configure_and_start_spark.sh -s <your-path> -t
```
