---
title: Running on Kestrel
---
# Kestrel Job Partitions and Scheduling Policies

*Learn about job partitions and policies for scheduling jobs on Kestrel.*

## Partitions

Kestrel nodes are associated with one or more partitions.  Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, per-node local scratch disk requirements, and whether graphics processing units (GPUs) are needed.

Jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**

The following table summarizes the partitions on Kestrel.

| Partition Name | Description   | Limits | Placement Condition |
| -------------- | ------------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes with each of the non-standard hardware configurations are available. The node-type distribution is: <br> - 2 Bigmem nodes <br> - 2 nodes with 1.7 TB NVMe <br> - 4 standard nodes <br> - **8 total nodes** | - 1 job with a <br>max of 2 nodes <br>per user <br>- 01:00:00 max walltime | ```-p debug``` <br>   or<br>   ```--partition=debug``` |
|```short```     |  Nodes that prefer jobs with walltimes <= 4 hours | No partition limit. <br> No limit per user. | ```--time <= 4:00:00```<br>```--mem <= 250000```<br> ```--tmp<=1700000 (256 nodes)```<br>  2106 nodes | 
| ```standard``` | Nodes that prefer jobs with walltimes <= 2 days | 2100 nodes total<br> 1050 nodes per user | ```--mem <= 250000```<br> ```--tmp<=1700000````|
|```bigmem```    | Nodes that have 2 TB of RAM with jobs of walltimes <= 2 days | 8 nodes total<br> 45 nodes per user | ```--mem > 2000000``` ```--time <= 2-00`` ```--tmp > 5800000``` |
|```bigmeml```    | Nodes that have 2 TB of RAM and jobs with walltimes >2 days  | 4 nodes total<br> 3 nodes per user | ```--mem > 180224``` | 
| ```long```     | Nodes that prefer jobs with walltimes > 2 days<br>*Maximum walltime of any job is 10 days*| 525 nodes total<br> 262 nodes per user|  ```--time <= 10-00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|





Use the option listed above on the ```srun```, ```sbatch```, or ```salloc``` command or in your job script to specify what resources your job requires.  Sample job scripts and the syntax for specifying the queue are available on the [sample job scripts page](./sample_sbatch.md).

## Job Scheduling Policies
The [Kestrel system configuration page](https://www.nrel.gov/hpc/kestrel-system-configuration.html) lists the four categories that Kestrel nodes exhibit based on their hardware features. No single user can have jobs running on more than half of the nodes from each hardware category. For example, the maximum quantity of data and analysis visualization (DAV) nodes a single job can use is 25.

Also learn how [jobs are prioritized](./eagle_job_priorities.md). 