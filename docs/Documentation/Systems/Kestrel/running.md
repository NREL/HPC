---
title: Running on Kestrel
---
# Kestrel Job Partitions and Scheduling Policies

*Learn about job partitions and policies for scheduling jobs on Kestrel.*

## Partitions

Kestrel nodes are associated with one or more partitions.  Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, and per-node local scratch disk requirements.

Jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**

The [Kestrel system configuration page](https://www.nrel.gov/hpc/kestrel-system-configuration.html) lists the four categories that Kestrel nodes exhibit based on their hardware features. 

The following table summarizes the partitions on Kestrel:


| Partition Name | Description   | Limits | Placement Condition |
| -------------- | ------------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes <br> with each of the non-standard <br> hardware configurations are available. <br> The node-type distribution is: <br> - 2 Bigmem nodes <br> - 2 nodes with 1.7 TB NVMe <br> - 4 standard nodes <br> - **8 total nodes** | 1 job with a <br>max of 2 nodes <br>per user <br> 01:00:00 max walltime | ```-p debug``` <br>   or<br>   ```--partition=debug``` |
|```short```     |  Nodes that prefer jobs with walltimes <= 4 hours | 2016 nodes total. <br> No limit per user. | ```--time <= 4:00:00```<br>```--mem <= 250000```<br> ```--tmp <= 1700000 (256 nodes)```| 
| ```standard``` | Nodes that prefer jobs with walltimes <= 2 days. | 2106 nodes total. <br> 1050 nodes per user. | ```--mem <= 250000```<br> ```--tmp <= 1700000```|
| ```long```     | Nodes that prefer jobs with walltimes > 2 days.<br>*Maximum walltime of any job is 10 days*| 525 nodes total<br> 262 nodes per user|  ```--time <= 10-00```<br>```--mem <= 250000```<br>```--tmp <= 1700000 (256 nodes)```|
|```bigmem```    | Nodes that have 2 TB of RAM and 5.8 TB NVMe local disk. | 8 nodes total<br> 4 nodes per user | ```--mem > 250000```<br> ```--time <= 2-00```<br>```--tmp > 1700000 ``` |
|```bigmeml```    | Bigmem nodes that prefer jobs with walltimes > 2 days.<br>*Maximum walltime of any job is 10 days*  | 4 nodes total<br> 3 nodes per user | ```--mem > 250000```<br>```--time > 2-00```<br>```--tmp > 1700000 ``` | 


Use the option listed above on the ```srun```, ```sbatch```, or ```salloc``` command or in your job script to specify what resources your job requires.  

!!! note
    For now, more information on Slurm and job submission script examples can be found under the [Eagle Running Jobs section](../Eagle/Running/index.md).
    

## Job Submission Recommendations

#### OpenMP

When running codes with OpenMP enabled, we recommend manually setting one of the following environment variables:

```

export OMP_PROC_BIND=spread # for non-intel built codes

export KMP_AFFINITY=balanced # for codes built with intel compilers

```
You may need to export these variables even if you are not running your job with threading, i.e., with `OMP_NUM_THREADS=1`

#### Scaling

Currently, some applications on Kestrel are not scaling with the expected performance. For these applications, we recommend:

1. Submitting jobs with the fewest number of nodes possible.

1. For hybrid MPI/OpenMP codes, requesting more threads per task than you tend to request on Eagle. This may yield performance improvements.
1. Building and running with Intel MPI or Cray MPICH, rather than OpenMPI.


