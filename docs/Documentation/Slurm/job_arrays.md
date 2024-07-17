---
layout: default
title: Job Arrays
has_children: false
---
# Job Arrays
Job arrays are typically used when a user wants to submit many similar jobs with different inputs. Job arrays are capable of submitting hundreds, and even thousands, of similar jobs together. Here, we will describe how to submit job arrays on Slurm. More details on job arrays can be found in the [Slurm documentation](https://slurm.schedmd.com/job_array.html).

An example of a job array submission script can be found in our [NREL HPC Slurm Examples](https://github.com/NREL/HPC/tree/master/slurm) directory. The job array example is titled [uselist.sh](https://github.com/NREL/HPC/blob/master/slurm/uselist.sh), and requires [doarray.py](https://github.com/NREL/HPC/blob/master/slurm/source/doarray.py) and [invertc.c](https://github.com/NREL/HPC/blob/master/slurm/source/invertc.c) from the source folder.

## SBATCH Directives for Job Arrays
In order to submit a job array to Slurm, the SBATCH directives at the top of your script or sbatch command line submission must contain the flag `--array=<ARRAY_VALS>`, where `ARRAY_VALS` is a list or range of numbers that will represent the index values of your job array. For example:

```
# SBATCH --array=0-12  # Submits a job array with index values between 0 and 12
...

# SBATCH --array=2,4,6,10  # Submits a job array with index values 2, 4, 6, and 10
...

# SBATCH --array=1-43:2  # Submits a job array with index values between 1 and 43 with a step size of 2
...

# SBATCH --array=1-25%5  # Submits a job array with index values between 1 and 25 and limits the number of simultaneously running tasks to 5
```

!!! note "Submitting Job Arrays on Kestrel"
    To ensure that your job array is running optimally, it is recommended that job arrays are submitted on the shared partition using `--partition=shared`. See more about shared partitions on Kestrel [here](/Documentation/Systems/Kestrel/running/#shared-node-partition).

## Job Control
Like standard slurm jobs, job arrays have a JOB_ID, which is stored in the environment variable `SLURM_ARRAY_JOB_ID`. The environment variable `SLURM_ARRAY_TASK_ID` will hold information about the index of the job array.

For example, if there is a job array in the queue, the output may look like this:

```
$ squeue
 JOBID   PARTITION     NAME     USER  ST  TIME NODES NODELIST
 45678_1  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_2  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_3  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_4  standard    array     user  R  0:13  1     x1007c0s0b0n1

```

Here, `SLURM_ARRAY_JOB_ID` is 45678. The number followed by the underscore in row is the `SLURM_ARRAY_TASK_ID`. This job is a job array that was submitted with `--array=1-4`.

Scontrol commands can be executed on entire job arrays or specific indices of a job array.
```
$ scontrol suspend 45678 
$ squeue
 JOBID   PARTITION     NAME     USER  ST  TIME NODES NODELIST
 45678_1  standard    array     user  S  0:13  1     x1007c0s0b0n1
 45678_2  standard    array     user  S  0:13  1     x1007c0s0b0n1
 45678_3  standard    array     user  S  0:13  1     x1007c0s0b0n1
 45678_4  standard    array     user  S  0:13  1     x1007c0s0b0n1

$ scontrol resume 45678
$ squeue
 JOBID   PARTITION     NAME     USER  ST  TIME NODES NODELIST
 45678_1  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_2  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_3  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_4  standard    array     user  R  0:13  1     x1007c0s0b0n1

```
```
$ scontrol suspend 45678_2 
$ squeue
 JOBID   PARTITION     NAME     USER  ST  TIME NODES NODELIST
 45678_1  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_2  standard    array     user  S  0:13  1     x1007c0s0b0n1
 45678_3  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_4  standard    array     user  R  0:13  1     x1007c0s0b0n1

$ scontrol resume 45678_2
$ squeue
 JOBID   PARTITION     NAME     USER  ST  TIME NODES NODELIST
 45678_1  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_2  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_3  standard    array     user  R  0:13  1     x1007c0s0b0n1
 45678_4  standard    array     user  R  0:13  1     x1007c0s0b0n1

```