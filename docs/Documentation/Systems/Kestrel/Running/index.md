---
title: Running on Kestrel
---
# Kestrel Job Partitions and Scheduling Policies

*Learn about job partitions and policies for scheduling jobs on Kestrel.*

## Kestrel Compute Nodes

There are two general types of compute nodes on Kestrel: CPU nodes and GPU nodes. These nodes can be further broken down into four categories, listed on the [Kestrel System Configuration Page](https://www.nrel.gov/hpc/kestrel-system-configuration.html).


### CPU Nodes
Standard CPU-based compute nodes on Kestrel have 104 cores and 240G of usable RAM. 256 of those nodes have a 1.7TB NVMe local disk. There are also 10 bigmem nodes with 2TB of RAM and 5.6TB NVMe local disk. Two racks of the CPU compute nodes have dual network interface cards (NICs) which may increase performance for certain types of multi-node jobs. 


### GPU Nodes
Kestrel has 132 GPU nodes with 4 NVIDIA H100 GPUs, each with 80GB memory. These have Dual socket AMD Genoa 64-core processors (128 cores total) with about 350G of usable RAM. The GPU nodes also have 3.4TB of NVMe local disk. 

!!! warning
    You should use a [login node](../index.md) that matches the architecture of the compute nodes that your jobs will be running on for compiling software and submitting jobs. 

### Using Node Local Storage


The majority of CPU nodes do not have local disk storage, but there are 256 nodes with fast local NVMe drives for temporary storage by jobs with high disk I/O requirements. To request nodes with local disk, use the `--tmp` option in your job submission script (e.g. `--tmp=1600000`). When your job is allocated nodes with local disk, the storage may then be accessed inside the job by using the `$TMPDIR` environment variable as the path. Be aware that on nodes without local disk, writing to `$TMPDIR` will consume RAM, reducing the available memory for running processes.  

Note that all of the Bigmem and H100 GPU nodes have real local disk. 


## Partitions

Kestrel nodes are associated with one or more partitions. Each partition is associated with one or more job characteristics, which include run time, per-node memory requirements, and per-node local scratch disk requirements.

Excluding the shared and debug partitions, jobs will be automatically routed to the appropriate partitions by Slurm based on node quantity, walltime, hardware features, and other aspects specified in the submission. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission.**.


The following table summarizes the partitions on Kestrel:

| Partition Name | Description   | Limits | Placement Condition |
| -------------- | ------------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes with each of the non-standard <br> hardware configurations are available. <br> The node-type distribution is: <br> - 2 bigmem nodes <br> - 2 nodes with 1.7 TB NVMe <br> - 4 standard nodes <br> - 2 GPU nodes (shared) <br> **10 total nodes** | - 1 job with a max of 2 nodes per user. <br> - 2 GPUs per user.<br> - 1/2 GPU node resources per user (Across 1-2 nodes). <br> - 01:00:00 max walltime. | ```-p debug``` <br>   or<br>   ```--partition=debug``` |
|```short```     |  Nodes that prefer jobs with walltimes <br> <= 4 hours. | 2016 nodes total. <br> No limit per user. | ```--time <= 4:00:00```<br>```--mem <= 246064```<br> ```--tmp <= 1700000 (256 nodes)```| 
| ```standard``` | Nodes that prefer jobs with walltimes <br> <= 2 days. | 2106 nodes total. <br> 1050 nodes per user. | ```--mem <= 246064```<br> ```--tmp <= 1700000```|
| ```long```     | Nodes that prefer jobs with walltimes > 2 days.<br>*Maximum walltime of any job is 10 days*| 525 nodes total.<br> 262 nodes per user.|  ```--time <= 10-00```<br>```--mem <= 246064```<br>```--tmp <= 1700000  (256 nodes)```|
|```bigmem```    | Nodes that have 2 TB of RAM and 5.6 TB NVMe local disk. | 8 nodes total.<br> 4 nodes per user. | ```--mem > 246064```<br> ```--time <= 2-00```<br>```--tmp > 1700000 ``` |
|```bigmeml```    | Bigmem nodes that prefer jobs with walltimes > 2 days.<br>*Maximum walltime of any job is 10 days.*  | 4 nodes total.<br> 3 nodes per user. | ```--mem > 246064```<br>```--time > 2-00```<br>```--tmp > 1700000 ``` | 
|```hbw```    | CPU compute nodes with dual network interface cards. | 512 nodes total.<br> 256 nodes per user. <br> Minimum 2 nodes per job. | ```-p hbw``` <br>```--time <= 10-00``` <br> ```--nodes >= 2```| 
| ```shared```|  Nodes that can be shared by multiple users and jobs. | 64 nodes total. <br> Half of partition per user. <br> 2 days max walltime.  | ```-p shared``` <br>   or<br>  ```--partition=shared```| 
| ```sharedl```|  Nodes that can be shared by multiple users and prefer jobs with walltimes > 2 days. | 16 nodes total. <br> 8 nodes per user. | ```-p sharedl``` <br>   or<br>  <nobr>```--partition=sharedl```</nobr>| 
| ```gpu-h100```|  Shareable GPU nodes with 4 NVIDIA H100 SXM 80GB Computational Accelerators. | 130 nodes total. <br> 65 nodes per user. | ```1 <= --gpus <= 4``` <br>  ```--time <= 2-00```| 
| ```gpu-h100s```|  Shareable GPU nodes that prefer jobs with walltimes <= 4 hours. | 130 nodes total. <br> 65 nodes per user. | ```1 <= --gpus <= 4``` <br>  ```--time <= 4:00:00```| 
| ```gpu-h100l```|  Shareable GPU nodes that prefer jobs with walltimes > 2 days. | 26 GPU nodes total. <br>  13 GPU nodes per user. | ```1 <= --gpus <= 4```<br> ```--time > 2-00```| 


Use the option listed above on the ```srun```, ```sbatch```, or ```salloc``` command or in your job script to specify what resources your job requires.  

For more information on running jobs and Slurm job scheduling, please see the [Slurm documentation section](../../../Slurm/index.md).

### Shared Node Partition 

Nodes in the shared partition can be shared by multiple users or jobs. This partition is intended for jobs that do not require a whole node.

!!! tip
    Testing at NREL has been done to evaluate the performance of VASP using shared nodes. Please see the [VASP page](../../../Applications/vasp.md#vasp-on-kestrel) for specific recommendations. 

#### Usage

Currently, there are 64 standard compute nodes available in the shared partition. These nodes have about 240G of usable RAM and 104 cores. By default, your job will be allocated about 1G of RAM per core requested. To change this amount, you can use the ```--mem``` or ```--mem-per-cpu``` flag in your job submission. To allocate all of the memory available on a node, use the `--mem=0` flag. 

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

### High Bandwidth Partition

In December 2024, Kestrel had two racks of CPU nodes reconfigured with an extra network interface card, which can greatly benefit communication-bound HPC software.
A NIC is a hardware component that enables inter-node (i.e., *network*) communication as multi-node jobs run. 
On Kestrel, most CPU nodes include a single NIC. Although having one NIC per node is acceptable for the majority of workflows run on Kestrel, it can lead to communication congestion 
when running multi-node applications that send significant amounts of data over Kestrel's network. When this issue is encountered, increasing the number of available NICs 
can alleviate such congestion during runtime. Some common examples of communication-bound HPC software are AMRWind and [LAMMPS](../../../Applications/lammps.md).

To request nodes with two NICs, specify `--partition=hbw` in your job submissions. Because the purpose of the high bandwidth nodes is to optimize communication in multi-node jobs, it is not permitted to submit single-node jobs to the `hbw` partition.
If you would like assistance with determining whether your workflow could benefit from running in the `hbw` partition, please reach out to [HPC-Help@nrel.gov](mailto:HPC-Help).

!!! info
    We'll be continuing to update documentation with use cases and recommendations for the dual NIC nodes, including specific examples on the AMRWind page. 


### GPU Jobs

Each GPU node has 4 NVIDIA H100 GPUs (80 GB), 128 CPU cores, and 350GB of useable RAM. All of the GPU nodes are shared. We highly recommend considering the use of partial GPU nodes if possible in order to efficiently use the GPU nodes and your AUs. 

To request use of a GPU, use the flag `--gpus=<quantity>` with sbatch, srun, or salloc, or add it as an `#SBATCH` directive in your sbatch submit script, where `<quantity>` is a number from 1 to 4. All of the GPU memory for each GPU allocated will be available to the job (80 GB per GPU).

**If your job will require more than the default 1 CPU core and 1G of CPU RAM per core allocated**, you must request the quantity of cores and/or RAM that you will need, by using additional flags such as `--ntasks=` or `--mem=`. To request all of the memory available on the GPU node, use `--mem=0`. 

The GPU nodes also have 3.4 TB of local disk space. Note that other jobs running on the same GPU node could also be using this space. Slurm is unable to divide this space to separate jobs on the same node like it does for memory or CPUs. If you need to ensure that your job has exclusive access to all of the disk space, you'll need to use the `--exclusive` flag to prevent the node from being shared with other jobs.

!!! warning
    A job with the ` --exclusive` flag will be allocated all of the CPUs and GPUs on a node, but is only allocated as much memory as requested. Use the flag `--mem=0` to request all of the CPU RAM on the node. 

#### GPU Debug Jobs

There are two shared GPU nodes available for debugging. To use them, specify `--partition=debug` in your job script. In addition to the limits for the `debug` partition, 1 job per user, up to 2 nodes per user, up to 1 hour of walltime, a single GPU job is also limited to half of a total GPU node's resources. This is equivalent to 64 CPU cores, 2 GPUs, and 180G of RAM, which can be spread across 1 or 2 nodes. Unlike the other GPU nodes, the GPU debug nodes can't be used exclusively, so the `--exclusive` flag can't be used for debug GPU jobs. 

## Allocation Unit (AU) Charges

The equation for calculating the AU cost of a job is:

`AU cost = (Walltime in hours * Number of Nodes * QoS Factor * Charge Factor)`

The CPU node charge factor is 10, and the GPU node charge factor is 100. 

On shared nodes (nodes in the `shared` partition and GPU nodes), the value for `Number of Nodes` can be a fraction of a node. This value will be calculated based on either the number of cores, amount of memory, or the number of GPUs (on GPU nodes), whichever is a greater percentage of the total of that resource available on the node.


???+ example "Example Job Cost Calculation - CPU shared "
    For example, if you request 123032M of RAM (half of the available RAM on the node), and 26 cores, you will be billed 5 AUs per node hour.

    ```bash
    # To determine the Number of Nodes value: 
    123032/246064 = 0.5

    26/104 = 0.25 

    Number of Nodes = 0.5

    # Final calculation

    1 hour walltime * 0.5 nodes * 1 QoS Factor * 10 Charge Factor = 5 AUs

    ```
???+ example "Example Job Cost Calculation - GPU "
    For example, if you request 270000M of RAM, 32 cores, and 2 GPUs you will be billed 75 AUs per node hour.

    ```bash
    # To determine the Number of Nodes value: 
    
    # CPU RAM
    270000/360000  0.75

    # CPU Cores 
    32/128 = 0.25 

    # GPUs
    2/4 = 0.5


    Number of Nodes = 0.75

    # Final calculation

    1 hour walltime * 0.75 nodes * 1 QoS Factor * 100 Charge Factor = 75 AUs

    ```

If a job requests the maximum amount of any resource type available on the node (CPUs, GPUs, RAM), it will be charged with the full charge factor (10 or 100).

## Performance Recommendations

Please see [this page](./performancerecs.md) for our most up-to-date performance recommendations on Kestrel.

