---
title: Running on Kestrel
---
# Kestrel Job Partitions and Scheduling Policies

*Learn about job partitions and policies for scheduling jobs on Kestrel.*

## Kestrel Compute Nodes

The [Kestrel system configuration page](https://www.nrel.gov/hpc/kestrel-system-configuration.html) lists the four categories that Kestrel nodes exhibit based on their hardware features. In summary, standard compute nodes on Kestrel have 104 cores and 248GB of usable RAM. 256 of those nodes have a 1.7 TB NVMe local disk. There are also 10 bigmem nodes with 2 TB of RAM and 5.8 TB NVMe local disk.

### GPU Nodes
Kestrel has 132 GPU nodes with 4 NVIDIA H100 GPUs, each with 80 GB memory. These have Dual socket AMD Genoa 64-core processors (128 cores total) with 360 GB of usable RAM. The GPU nodes also have 3.25 TB of NVMe local disk. 



### Using Node Local Storage


To use local disk on the nodes that have it available, use the `$TMPDIR` environment variable. On nodes without local disk, writing here will consume RAM. To request nodes with local disk, use the `--tmp` option in your job submission script. (e.g. `--tmp=1600000`). Note that all of the Bigmem and H100 GPU nodes have real local disk. 


## Partitions

Kestrel nodes are associated with one or more partitions. Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, and per-node local scratch disk requirements.

Excluding the shared and debug partitions, jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**.


The following table summarizes the partitions on Kestrel:


| Partition Name | Description   | Limits | Placement Condition |
| -------------- | ------------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes with each of the non-standard <br> hardware configurations are available. <br> The node-type distribution is: <br> - 2 bigmem nodes <br> - 2 nodes with 1.7 TB NVMe <br> - 4 standard nodes <br> - 2 GPU nodes <br> **10 total nodes** | 1 job with a max of 2 nodes per user*. <br> 01:00:00 max walltime. | ```-p debug``` <br>   or<br>   ```--partition=debug``` |
|```short```     |  Nodes that prefer jobs with walltimes <br> <= 4 hours. | 2016 nodes total. <br> No limit per user. | ```--time <= 4:00:00```<br>```--mem <= 248000```<br> ```--tmp <= 1700000 (256 nodes)```| 
| ```standard``` | Nodes that prefer jobs with walltimes <br> <= 2 days. | 2106 nodes total. <br> 1050 nodes per user. | ```--mem <= 248000```<br> ```--tmp <= 1700000```|
| ```long```     | Nodes that prefer jobs with walltimes > 2 days.<br>*Maximum walltime of any job is 10 days*| 525 nodes total.<br> 262 nodes per user.|  ```--time <= 10-00```<br>```--mem <= 248000```<br>```--tmp <= 1700000  (256 nodes)```|
|```bigmem```    | Nodes that have 2 TB of RAM and 5.8 TB NVMe local disk. | 8 nodes total.<br> 4 nodes per user. | ```--mem > 248000```<br> ```--time <= 2-00```<br>```--tmp > 1700000 ``` |
|```bigmeml```    | Bigmem nodes that prefer jobs with walltimes > 2 days.<br>*Maximum walltime of any job is 10 days.*  | 4 nodes total.<br> 3 nodes per user. | ```--mem > 248000```<br>```--time > 2-00```<br>```--tmp > 1700000 ``` | 
| ```shared```|  Nodes that can be shared by multiple users and jobs. | 64 nodes total. <br> No limit per user. <br> 2 days max walltime.  | ```-p shared``` <br>   or<br>  ```--partition=shared```| 
| ```sharedl```|  Nodes that can be shared by multiple users and prefer jobs with walltimes > 2 days. | 16 nodes total. <br> 8 nodes per user. | ```-p sharedl``` <br>   or<br>  <nobr>```--partition=sharedl```</nobr>| 
| ```gpu-h100```|  Nodes with 4 NVIDIA H100 SXM 80GB Computational Accelerators. | 130 nodes total. <br> 65 nodes per user. | ```1 <= --gpus <= 4``` <br>  ```--time <= 2-00```| 
| ```gpu-h100l```|  GPU nodes that prefer jobs with walltimes > 2 days. | 13 nodes total. <br>  7 nodes per user. | ```1 <= --gpus <= 4```<br> ```--time > 2-00```| 

*GPU jobs in the debug partition are limited to 2 GPUs...

Use the option listed above on the ```srun```, ```sbatch```, or ```salloc``` command or in your job script to specify what resources your job requires.  

For more information on running jobs and Slurm job scheduling, please see the [Slurm documentation section](../../Slurm/index.md).

### Shared Node Partition 

Nodes in the shared partition can be shared by multiple users or jobs. This partition is intended for jobs that do not require a whole node.

!!! tip
    Testing at NREL has been done to evaluate the performance of VASP using shared nodes. Please see the [VASP page](../../Applications/vasp.md#vasp-on-kestrel) for specific recommendations. 

#### Usage

Currently, there are 64 standard compute nodes available in the shared partition. These nodes have 248GB of usable RAM and 104 cores. By default, your job will be allocated 1.024GB of RAM per core requested To change this amount, you can use the ```--mem``` or ```--mem-per-cpu``` flag in your job submission. 

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


### GPU Jobs

Each GPU node has 4 NVIDIA H100 GPUs (80 GB), 128 CPU cores, and 360GB of useable RAM. All of the GPU nodes are shared. We highly recommend considering using partial GPU nodes if possible in order to efficiently use the GPU nodes and your AUs. 

To request use of a GPU, use the flag `--gpus=<quantity>` with sbatch, srun, or salloc, or add it as an `#SBATCH` directive in your sbatch submit script, where <quantity> is a number from 1 to 4. All of the GPU memory for each GPU allocated will be available to the job (80GB per GPU).

If your job will require more than the default 1 CPU core and 2300MB of CPU RAM per core, you must request the quantity of cores and/or RAM that you will need, by using additional flags such as `--ntasks=` or `--mem=`.

The GPU nodes also have 3250000MB space of local disk. Note that other jobs running on the same GPU could also be using this space. If you need to ensure that your job has all of the disk space, you'll need to request the whole GPU node (--exclusive). Tmp isn't a requestable resources. Slurm isn't able to allocate this space based on job

Using --exclusive, or requesting the entirity of one or more of the resources (gpu, ram, cpu), will allocate the job the entire GPU node. 

## Allocation Unit (AU) Charges

The equation for calculating the AU cost of a job is:

`AU cost = (Walltime in hours * Number of Nodes * QoS Factor * Charge Factor)`

The CPU node charge factor is 10, and the GPU node charge factor is 100. 

On shared nodes (nodes in the shared partition and GPU nodes), the value for `Number of Nodes` can be a fraction of a node. This value will be calculated based on either the number of cores, amount of memory, or the number of GPUs (on GPU nodes), whichever is a greater percentage of the total of that resource available on the node.


The highest quantity of resource requested will determine the total AU charge.
To summarize:

CPU nodes in the shared partition have 

1 GPU = 25% of total cores (24/128) = 25% of total RAM (256GB/1TB) = 25% of a node

???+ example "Example Job Cost Calculation - CPU "
    For example, if you request 124GB of RAM (half of the available RAM on the node), and 26 cores, you will be billed 5 AUs per node hour.

    ```bash
    # To determine the Number of Nodes value: 
    124/248 = 0.5

    26/104 = 0.25 

    Number of Nodes = 0.5

    # Final calculation

    1 hour walltime * 0.5 nodes * 1 QoS Factor * 10 Charge Factor = 5 AUs

    ```
???+ example "Example Job Cost Calculation - GPU "
    For example, if you request 270GB of RAM, and 32 cores, and 2 GPUs you will be billed 75 AUs per node hour.

    ```bash
    # To determine the Number of Nodes value: 
    
    # CPU RAM
    270/360 = 0.75

    # CPU Cores 
    32/128 = 0.25 

    # GPUs
    2/4 = 0.5


    Number of Nodes = 0.75

    # Final calculation

    1 hour walltime * 0.75 nodes * 1 QoS Factor * 100 Charge Factor = 75 AUs

    ```

Using --exclusive or requesting all of one of the resources (ram, cpu, gpu) will charge you the full AU cost. 


    
## Performance Recommendations

Please see [this page](../eagle_to_kestrel_transition.md#5-performance-recommendations) for our most up-to-date performance recommendations on Kestrel.


