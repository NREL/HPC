---
title: Running on Kestrel
---
# Kestrel Job Partitions and Scheduling Policies

*Learn about job partitions and policies for scheduling jobs on Kestrel.*

## Partitions

Kestrel nodes are associated with one or more partitions.  Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, and per-node local scratch disk requirements.

Excluding the shared and debug partitions, jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**.

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
| ```shared```|  Nodes that can be shared by multiple users and jobs. Walltime <= 2 days. | 32 nodes total. <br> No limit per user. | ```-p shared``` <br>   or<br>  ```--partition=shared```| 

Use the option listed above on the ```srun```, ```sbatch```, or ```salloc``` command or in your job script to specify what resources your job requires.  

For more information on running jobs and Slurm job scheduling, please see the [Slurm documentation section](../../Slurm/index.md).

### Shared Node Partition 

Unlike the other partitions, nodes in the shared partition can be shared by multiple users or jobs. This partition is intended for jobs that do not require a whole node.

!!! tip
    Testing at NREL has been done to evaluate the performance of VASP using shared nodes. Please see the [VASP page](../../Applications/vasp.md#vasp-on-kestrel) for specific recommendations. 

#### Usage

Currently, there are 32 standard compute nodes available in the shared partition. These nodes have 250GB of usable RAM and 104 cores. By default, your job will be allocated 1.024GB of RAM per core requested To change this amount, you can use the ```--mem``` or ```--mem-per-cpu``` flag in your job submission. 

??? info "Sample batch script for a job in the shared partition"
    ```
    #!/bin/bash
    #SBATCH --nodes=1 
    #SBATCH --partition=shared         
    #SBATCH --time=2:00:00    
    #SBATCH --ntasks=26 # CPUs requested for job 
    #SBATCH --mem-per-cpu=2000 # Request 2GB per core.
    #SBATCH --account=<allocation handle>

    cd /scratch/$USER 
    srun ./my_progam # Use your application's commands here  
    ```


#### Accounting

The equation for calculating the AU cost of a job is:

`AU cost = (Walltime in hours * Number of Nodes * QoS Factor * Charge Factor)`

In the shared node partition, the value for `Number of Nodes` can be a fraction of a node. This value will be calculated based on either the amount of cores or the amount of memory requested, whichever is a greater percentage of the total of that resource available on the node.

???+ example "Example Job Cost Calculation"
    For example, if you request 125GB of RAM (half of the available RAM on the node), and 26 cores, you will be billed 5 AUs per node hour.
    ```
    # To determine the Number of Nodes value: 
    125/250 = 0.5

    26/104 = 0.25 

    Number of Nodes = 0.5

    # Final calculation

    1 hour walltime * 0.5 nodes * 1 QoS Factor * 10 Charge Factor = 5 AUs
    ```
    
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


