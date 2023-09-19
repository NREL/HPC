# Transitioning from Eagle to Kestrel

*This page summarizes key points to know for getting started on Kestrel. Documentation will continue to be posted in the Kestrel section and updated on other pages.*

## Accessing Kestrel

### For NREL Employees:

To access Kestrel, connect via ssh to `kestrel.hpc.nrel.gov` from the NREL network. 

### Extern
To access Kestrel, login into BLANK with your usual NREL HPC credentials. More details are provided here: 

Please see (page)

## System Comparison 

|                    |     Eagle     |     Kestrel     |
| :------------------: | :-------------: | :-------------: |
| Peak Performance |       8 Pflops peak       |   44 Pflops peak            |
| Processors |       Intel Xeon-Gold Skylake<br> **18 cores**       |      Intel Sapphire Rapids<br> **52 cores**        |
| Nodes  |     2,114 nodes, 2 processors/node<br>​ **36 cores/node**       |    2,304 nodes, 2 processors/node<br>​ **104 cores/node**       |
| Data Storage   |       **14 PB** Lustre       |      **95 PB** Lustre         |

## Running Jobs

Like Eagle, Kestrel uses Slurm for job scheduling. As on Eagle, jobs will be automatically routed to the appropriate partitions by Slurm based on resources requested. Jobs will have access to the largest number of nodes, thus shortest wait, **if the partition is not specified during job submission**.

Currently, nodes are allocated exclusively. A shared node parition is being tested. 

The parititions on Kestrel are similar to Eagle's. There are no gpu paritions since GPUs are not yet available, and there is no `bigscratch` partition. If your job needs local disk space, use the `--tmp` option in your job submission script and Slurm will automatically place your job on a node with sufficient resources. 

### Compute Nodes
Dual socket Intel Xeon Sapphire Rapids (52-core) processors
Kestrel currently has no GPU nodes. They are expected to be available in Q2 of FY24. 

Compute nodes on Kestrel
The compute nodes on Kestrel have 104 cores per node. There are 2144 standard compute nodes with 256GB RAM. 256 of those nodes have 1.75TB local disk space. 

There are 10 bigmem nodes, with 2TB of RAM and 5.8TB local disk space. 





| Partition Name | Description   | Limits | Placement Condition |
| -------------- | ------------- | ------ | ------------------- | 
| ```debug```    | Nodes dedicated to developing and <br> troubleshooting jobs. Debug nodes <br> with each of the non-standard <br> hardware configurations are available. <br> The node-type distribution is: <br> - 4 GPU nodes <br> - 2 Bigmem nodes <br>- 7 standard nodes <br> - **13 total nodes** | 1 job with a <br>max of 2 nodes <br>per user <br> 01:00:00 max walltime | ```-p debug``` <br>   or<br>   ```--partition=debug``` |
|```short```     |  Nodes that prefer jobs with walltimes <= 4 hours | No partition limit. <br> No limit per user. | ```--time <= 4:00:00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|
| ```standard``` | Nodes that prefer jobs with walltimes <= 2 days | 2100 nodes total<br> 1050 nodes per user | ```--time <= 2-00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|
| ```long```     | Nodes that prefer jobs with walltimes > 2 days<br>*Maximum walltime of any job is 10 days*| 525 nodes total<br> 262 nodes per user|  ```--time <= 10-00```<br>```--mem <= 85248   (1800 nodes)```<br>```--mem <= 180224 (720 nodes)```|
|```bigmem```    | Nodes that have 2TB GB of RAM | 10 nodes total | ```--mem > 180224``` |




### Notable Job Submission Recommendation Changes:


## File System

Kestrel has a **95 PB** ClusterStor Lustre file system. 

ScratchFS uses a Lustre file system in a hybrid flash-disk configuration providing a total of 27 petabytes (PB) of capacity with 354 gigabytes (GB)/s of IOR bandwidth. ProjectFS will provide 68 PB of capacity with 200 GB/s of IOR bandwidth. So, unlike on Eagle, running jobs out of /scratch will be more performant than /projects. So, we advise running jobs out of /scratch and moving data to /projects for long term storage. Like on Eagle, /scratch will have a 28 day purge policy with no exceptions. 

The Home File System (HFS) on Kestrel is part of the ClusterStor used for PFS, providing highly reliable storage for user home directories and NREL-specific software. HFS will provide 1.2 PB of capacity. Snapshots of files on the HFS will be available up to 30 days after change/deletion. /home directories have a quota of 50 GB. 



## Data Transfer

To transfer small batches of data, use `rsync` or `scp`. 

Globus is expected to be available soon. Please contact [HPC-Help@nrel.gov](mailto://hpc-help@nrel.gov) if you need help with transferring data to Kestrel. 

## Jupyterhub

Jupyterhub is not yet available. 

## Environments 

Please see the [Kestrel Environments section](./Environments/index.md) for detailed information on Kestrel's modules and programming environments. 

## Additional Resources

* [Kestrel System Configuration](https://www.nrel.gov/hpc/kestrel-system-configuration.html)
* A collection of sample makefiles, source codes, and scripts for Kestrel can be found in the [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel). 
<!--TODO: Post Training Slides PDF once complete --> 

## Contributions

The [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel) is open for contributions of examples, scripts, and other resources that would benefit the user community. To contribute, please open a Pull Request or contact [HPC-Help@nrel.gov](mailto://hpc-help@nrel.gov). To recommend topics to be covered, please open an [issue](https://github.com/NREL/HPC/issues) in the repository.

