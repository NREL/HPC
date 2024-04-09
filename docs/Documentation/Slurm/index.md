---
layout: default
title: Slurm Job Scheduler
has_children: true
hide:
 - toc
---

## Schedule Your Computational Work with Slurm

[Slurm](https://slurm.schedmd.com/) is the job scheduler and workload manager used by the HPC clusters hosted at NREL. 

A **job** contains a list of required consumable resources (such as nodes), a list of job constraints (when, where and how the job should run), and an execution environment, which includes things like an executable, input and output files. All computational work on an HPC cluster should generally be contained in a job.

There are two key types of jobs:

* [Batch jobs](./batch_jobs.md) are unattended scripts that launch programs to complete computational work. Batch jobs are placed in a queue and launched at a future time and date, determined by the priority of the job. Batch jobs are submitted to the queue using the `sbatch` command. 

* [Interactive jobs](./interactive_jobs.md) provide a shell prompt on a compute node and allow for software to be run that requires keyboard input from the user. The `salloc` and `srun` commands can be used to start an interactive job.

Most computational work is typically submitted as a batch script and queued for later automatic execution. Results from standard output and/or standard error will be stored in a file or files by Slurm (this behavior is customizable in your sbatch script.) Your software may or may not also produce its own output files.

Please see the navigation bar on the left under the Slurm Job Scheduling section for more information about how to submit a job.




