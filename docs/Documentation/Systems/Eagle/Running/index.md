---
title: Running on Eagle
---

# Running Jobs on the Eagle System

*Learn about running jobs on the Eagle high-performance computing (HPC) system.*

!!! note "Running Different Types of Jobs"
    * [Batch Jobs](./batch_jobs.md)
    * [Interactive Jobs](./interactive_jobs.md)
    * [Multiple Sub-Jobs](./multiple_sub_jobs.md)

## Job Scheduling and Management
To allow multiple users to share the system, Eagle uses the Slurm workload manager/job scheduler and resource manager. Slurm has commands for job submission, job monitoring, and job control (hold, delete, and resource request modification).

A **"job"** contains a list of required consumable resources (such as nodes), a list of job constraints (when, where and how the job should run), and an execution environment, which includes things like an executable, input and output files.

Both [interactive jobs](./interactive_jobs.md) (*i.e.*, where you are given a shell prompt on one of possibly several assigned compute nodes) and regular [batch jobs](./batch_jobs.md) are supported.

At present, compute nodes are scheduled so that each active job has exclusive access to its assigned nodes.

To run a job on Eagle, you must have a [project resource allocation](https://www.nrel.gov/hpc/resource-allocations.html).

Each project has a **project handle** associated with it, which was specified in the project request document. Jobs submitted without a valid project handle will be rejected with an error message.  Please note that this project identifier is referred to as an **allocation handle** in error messages and as an **account string** in system man pages.  The project handle may be included with the `-A` option either on the command line or within the batch script.  After usage exceeds the node hour allocation for a project, jobs will run at very low priority.

## Submitting Jobs
You can submit jobs using one of `sbatch`, `salloc`, or `srun`. Below are some nuances between these commands:

* `sbatch` and `salloc` both request resources from the system (and thus, must wait in the job queue for the appropriate resources); whereas srun is what actually executes commands across the allocated nodes, serving as a generic wrapper for various MPI interfaces and managing parallel task distribution.
* `salloc` is interactive and blocking, meaning your shell session will wait until the resources are granted, and you will be able to interact directly with the compute node(s) via the command line. The output of any executables will print directly to your terminal session.
* `sbatc`h` is the "background" analog to salloc, meaning your executable will run once the resources are allocated independent of your terminal session. Output from any executables you submit will be captured into output files (the default directory for these is where you launch the sbatch command for that job.)
* If you use `srun` outside of a job, it will first invoke `salloc` to get a resource allocation. If you use `srun` within a job, this constitutes a "job step" and parallelizes the given task(s), the distribution of which can be configured across the nodes with a multitude of argument flags such as `--ntasks-per-node`.

Some example job submissions:
```
sbatch -A <project-handle> -t 5:00:00 my_job

salloc -A <project-handle> -t 5

srun -A <project-handle> -t 15 -N 6 --pty $SHELL         # An alternative to salloc
```


