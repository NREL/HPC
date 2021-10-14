---
title: May 2021 Monthly Update
data: 2021-05-05
layout: default
brief: Slurm Fairshare, Queue Times, Advanced Jupyter Workshop
---

# Slurm Fairshare Refresher
FY21 saw the introduction of the "fairshare" priority algorithm in Eagle's job scheduler, Slurm. Queue times have been high during the Q2-Q3 rush and we've received some questions, so here's a quick refresher on Fairshare and what it means in regards to job scheduling.

The fairshare algorithm is a part of the Slurm "multi-factor priority" plugin that determines when a job should run. This algorithm is designed to help moderate queue usage by promoting jobs from under-utilized allocations, while over-utilized allocations get shifted towards CPU time that would otherwise be idle. The base fairshare value for an allocation is determined by the number of AUs allocated to a project, and is currently re-calculated on a quarterly basis. Every job that runs will affect the fairshare value, reducing the priority of future jobs. Larger jobs will have a larger impact, running smaller jobs will have less of an impact. The effects of any job on fairshare value will reduce by half every two weeks. And most importantly, fairshare only accounts for about half of job priority calculations--the rest relies on other factors, including the job's size, QOS setting, and partition.

# Queue Times
The allocation year transitioned from Q2 to Q3 on April 1st. The job queue leading up to the end of Q2 saw a very large spike in jobs submitted, and queue depth (job wait time) rose accordingly. A few projects saw some effect of fairshare, but much of the pressure came from over a third of all jobs being submitted as qos=high. Because of the large surge in jobs submitted, interactions with fairshare and a few projects that have used up their allocation we have been analyzing the scheduling algorithms. Based on some recommendations from SchedMD and internal analysis we have made a few adjustments to the slurm configuration. Those changes thus far appear to have alleviated some of the pressure on the queues as well as a reduction in the number of jobs submitted with qos=high.

# Advanced Jupyter workshop (10am May 13th, 2021)
Beyond the basics: this advanced Jupyter workshop will survey topics which enable you to get more out of your interactive notebooks. It will build on the recent Intro to Jupyter workshop and introduce additional Magic commands. Interacting with Slurm from a notebook will also be covered, and how this can be used to achieve multi-node parallelism. Additional topics include utilizing GPUs from a notebook, and parameterized notebook execution with Papermill.
