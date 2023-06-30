---
layout: default
title: Running Batch Jobs
has_children: false
---

# Running Batch Jobs

## Job Scheduling and Management
Batch jobs are run by submitting a job script to the scheduler. The job script contains the commands needed to set up your environment and run your application. (This is an "unattended" run, with results written to a file for later access.)

Once submitted, the scheduler will insert your job script into the queue to be run at some point in the future, based on priority.

To submit jobs on Eagle, the Slurm sbatch command should be used:

`$ sbatch --account=<project-handle> <batch_script>`

Scripts and program executables may reside in any file system, but input and output files should be read from or written to the /scratch file system mount. /scratch uses the Lustre filesystem which is designed to utilize the parallelized networking fabric that exists between Eagle nodes, and will result in much higher throughput on file manipulations.

Arguments to sbatch may be used to specify resource limits such as job duration (referred to as "walltime"), number of nodes, etc., as well as what hardware features you want your job to run with. These can also be supplied within the script itself by placing #SBATCH comment directives within the file. 

## Required Flags 

| Parameter    | Flag              | Example                  |  Explanation   |
| -----------  | ----------------- | ------------------------ |  ------------------ |
| Project handle | `--account`, `-A` | `--account=<handle>` or `-A <handle>` | Project handles are provided by HPC Operations at the beginning of an allocation cycle.|
 Maximum Job Duration (walltime) | `--time`, `-t` | `--time=1-12:05:50` <br>(1 day, 12 hours, 5 minutes, and 50 seconds) <br> or <br> `-t5` (5 minutes) | Recognized Time Formats: <br> `<days>-<hours>` <br> `<days>-<hours>:<min>` <br> `<days>-<hours>:<min>:<sec>` <br> `<hours>:<min>:<sec>` <br> `<min>:<sec>` <br> `<min>`|

## Resource Request Descriptions

| Parameter    | Flag              | Example                  |  Explanation   |
| -----------  | ----------------- | ------------------------ |  ------------------ |
| Nodes, Tasks, MPI Ranks | `--nodes` or `-N` <br> `--ntasks` or `-n` <br> `--ntasks-per-node` | `--nodes=20` <br> `--ntasks=40` <br> `--ntasks-per-node=20` | if `ntasks` is specified, it is important to indicate the number of nodes request as well. This helps with scheduling jobs on the fewest possible Ecells (racks) required for the job. <br><br> The maximum number of tasks that can be assigned per node is equal to the CPU (core) count of the node.|
| Memory  | `--mem`<br> `--mem-per-cpu` | `--mem=50000` | Memory per node <br> memory per task/MPI rank|
| Local disk (/tmp/scratch) | `--tmp` | `--tmp=10TB`<br>`--tmp=100GB`<br>`--tmp=1000000`| Request /tmp/scratch space in megabytes (default), GB, or TB.|
| GPUs  | `--gres:gpu` | `--gres:gpu:2` | Requests 2 GPUs. See system information for total number of GPUs.|

## Job Management and Output

| Parameter    | Flag              | Example                  |  Explanation   |
| -----------  | ----------------- | ------------------------ |  ------------------ |
| High priority | `--qos` | `--qos=high` | High-priority jobs will take precedence in the queue. *Note*: There is an AU penalty of *2X* for high-priority jobs.|
| Dependencies | `--dependency` | `--dependency=<condition>:<job_id>` <br><br>Conditions:<br><br>`after`<br>`afterany`<br>`afternotok`<br>`afterok`<br>`singleton` | You can submit jobs that will wait until a condition is met before running. <br><br><br>Conditions:<br><br>After the listed jobs have started<br>After the listed jobs have finished<br>After the listed jobs have failed<br>After the listed jobs returne xit code 0<br>After all existing jobs with the same name and user have ended|
| Job Name | `--job-name` | `--job-name=myjob` | A short, descriptive job name for easier identification in the queue.|
| Email notifications | `--mail-user` | `--mail-user=my.email@nrel.gov`<br>`--mail=type=ALL` | Slurm will send updates on job status change. Type can be specified with `--mail-type` as BEGIN, END, FAIL, or ALL.|
| Output | `--output`<br><br>`--error` | `--output=job_stdout`<br><br>`--output=job_stderr` | Defaults to `slurm-<jobid>.out`<br><br>Defaults to `slurm-<jobid>.out` (same file as stdout)<br><br> stdout and stderr will be written to the same file unless specified otherwise|


## Commonly Used Slurm Environment Variables

| Parameter        | Semantic Value    | Sample Value             |
| ---------------- | ----------------- | ------------------------ |
| `$LOCAL_SCRATCH` | Absolute directory path for local-only disk space per node. This should always be /tmp/scratch for compute nodes.| `/tmp/scratch`|
| `$SLURM_CLUSTER_NAME` | The cluster name as per the master configuration in Slurm. Identical to `$NREL_CLUSTER`. | `eagle`|
| `$SLURM_CPUS_ON_NODE` | Quantity of CPUs per compute node. | `104` |
| `$SLURMD_NODENAME` | Slurm name of the node on which the variable is evaluated. Matches hostname. | `r4i2n3`|
| `$SLURMD_JOB_ACCOUNT` | The Slurm account used to submit the job. Matches the project handle. | `csc000` |
| `$SLURM_JOB_CPUS_PER_NODE` | Contains value of `--cpus-per-node`, if specified. Should be equal or less than `$SLURM_CPUS_ON_NODE`. | 104| 
| `$SLURM_JOBID` or `$SLURM_JOB_ID` | Job ID assigned to the job. | 521837 |
| `$SLURM_JOB_NAME` | The assigned name of the job, or the command run if no name was assigned. | bash |
| `$SLURM_JOB_NODELIST` or `$SLURM_NODELIST` | Hostnames of all nodes assigned to the job, in Slurm syntax. | `r4i2n[1,3-6]` | 
| `$SLURM_JOB_NUM_NODES` or `$SLURM_NNODES` | Quantity of nodes assigned to the job. | 5 | 
| `$SLURM_JOB_PARTITION` | The scheduler partition the job is assigned to. | short |
| `$SLURM_JOB_QOS` | The Quality of Service the job is assigned to. | high |
| `$SLURM_NODEID` | A unique index value for each node of the job, ranging from 0 to `$SLURM_NNODES`. | 0 |
| `$SLURM_STEP_ID` or `$SLURM_STEPID` | Within a job, sequential srun commands are called "steps". Each srun increments this variable, giving each step a unique index nmber. This may be helpful for debugging, when seeking which step a job fails at. | 0 | 
| `$SLURM_STEP_NODELIST` | Within a job, `srun` calls can contain differing specifications of how many nodes should be used for the step. If your job requests 5 total nodes and you used `srun --nodes=3`, this variable would contain the list of the 3 nodes that participated in this job step. | `r4i2n[2-4]` |
| `$SLURM_STEP_NUM_NODES` | Returns the quantity of nodes requested for the job step (see entry on `$SLURM_STEP_NODELIST`.) | 3 |
| `$SLURM_STEP_NUM_TASKS` | Returns the quantity of tasks requested to be executed in the job step. Defaults to the task quantity of the job request. | 1 |
| `$SLURM_STEP_TASKS_PER_NODE` | Contains the value specified by `--tasks-per-node` in the job step. Defaults to the tasks-per-node of the job request. | 1 |
| `$SLURM_SUBMIT_DIR` | Contains the absolute path of the directory the job was submitted from. | `/projects/csc000` |
| `$SLURM_SUBMIT_HOST` | The hostname of the system from which the job was submitted. Should always be a login node. | el1 | 
| `$SLURM_TASKS_PER_NODE` | Contained the value specified by `--tasks-per-node` in the job request. | 1 |