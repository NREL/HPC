# Eagle Job Partitions and Scheduling Policies
*Learn about job partitions and policies for scheduling jobs on Eagle.*

## Partitions

Eagle nodes are associated with one or more partitions.  Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, per-node local scratch disk requirements, and whether graphics processing units (GPUs) are needed.

Jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**

The following table summarizes the partitions on Eagle.

| Partition Name | Description   | Limits | Placement Condition |
| -------------- | ------------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes <br> with each of the non-standard <br> hardware configurations are available. <br> The node-type distribution is: <br> - 4 GPU nodes <br> - 2 Bigmem nodes <br>- 7 standard nodes <br> - **13 total nodes** | 1 job with a <br>max of 2 nodes <br>per user <br> 01:00:00 max walltime | ```-p debug``` <br>   or<br>   ```--partition=debug``` |
|```short```     |  Nodes that prefer jobs with walltimes <= 4 hours | No partition limit. <br> No limit per user. | ```--time <= 4:00:00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|
| ```standard``` | Nodes that prefer jobs with walltimes <= 2 days | 2100 nodes total<br> 1050 nodes per user | ```--time <= 2-00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|
| ```long```     | Nodes that prefer jobs with walltimes > 2 days<br>*Maximum walltime of any job is 10 days*| 525 nodes total<br> 262 nodes per user|  ```--time <= 10-00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|
|```bigmem```    | Nodes that have 768 GB of RAM | 90 nodes total<br> 45 nodes per user | ```--mem > 180224``` |
|```bigscratch```| Nodes that each have larger /tmp/scratch mounts (24 TB SSD) for<br> per-node large-data tasks | 20 nodes total<br> 10 nodes per user |```--tmp > 1500000```|
|```gpu```       | Nodes with dual NVIDIA Tesla V100 PCIe <br> 16 GB Computational Accelerators for GPU-based software | 20 nodes total<br> 10 nodes per user<br> 2 GPUs per node | ```--gres=gpu:1 (1 per node)```<br>```--gres=gpu:2 (2 per node)```<br>```--timelimit <= 2 days```|
|```gpul```      | Nodes with dual NVIDIA Tesla V100 PCIe <br> 16 GB Computational Accelerators for GPU-based software | 8 nodes <br> 2 nodes per user<br> 2 GPUs per node | ```--gres=gpu:1 (1 per node)```<br>```--gres=gpu:2 (2 per node)```<br>```--timelimit > 2 days```|

Use the option listed above on the ```srun```, ```sbatch```, or ```salloc``` command or in your job script to specify what resources your job requires.  More details regarding these commands and how to write an sbatch script are available in the [Slurm Job Scheduler](/Documentation/Slurm/) section.

## Job Scheduling Policies
The [system configuration page](https://www.nrel.gov/hpc/eagle-system-configuration.html) lists the four categories that Eagle nodes exhibit based on their hardware features. No single user can have jobs running on more than half of the nodes from each hardware category. For example, the maximum quantity of data and analysis visualization (DAV) nodes a single job can use is 25.

Also learn how [jobs are prioritized](./eagle_job_priorities.md). 

