# Transitioning from Eagle to Kestrel

*This page summarizes key points to know for getting started on Kestrel. Documentation will continue to be posted in the Kestrel section and updated on other pages.*

## Accessing Kestrel
Access to Kestrel requires an NREL HPC account and access to a project allocation on Kestrel. 

###NREL Employees:

To access Kestrel, connect via ssh to `kestrel.hpc.nrel.gov` from the NREL network. 

### External Collaborators:
There are currently no external-facing login nodes on Kestrel. You will need to connect to the [SSH gateway](https://www.nrel.gov/hpc/ssh-gateway-connection.html) or [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) and then ssh to Kestrel as directed above. 

## System Comparison 

|                    |     Eagle     |     Kestrel     |
| :------------------: | :-------------: | :-------------: |
| Peak Performance |       8 Pflops peak       |   44 Pflops peak            |
| Processors |       Intel Xeon-Gold Skylake<br> **18 cores**       |      Intel Sapphire Rapids<br> **52 cores**        |
| Nodes  |     2,114 nodes, 2 processors/node<br>​ **36 cores/node**       |    2,304 nodes, 2 processors/node<br>​ **104 cores/node**       |
| Data Storage   |       **14 PB** Lustre       |      **95 PB** Lustre         |

## Running Jobs

Kestrel uses Slurm for job scheduling. As on Eagle, jobs will be automatically routed to the appropriate partitions by Slurm based on resources requested. Jobs will have access to the largest number of potential nodes to run on, thus the shortest wait, **if the partition is not specified during job submission**.

Currently, nodes are allocated exclusively. A shared node parition is being tested. 

The partitions on Kestrel are similar to Eagle's. There are no `gpu` partitions since GPUs are not yet available, and there is no `bigscratch` partition. If your job needs local disk space, use the `--tmp` option in your job submission script and Slurm will automatically place your job on a node with sufficient resources. 

### Compute Nodes
Kestrel compute nodes have 104 cores per node. There are 2144 standard compute nodes with **256GB RAM**. 256 of those nodes have **1.75TB local disk space**. There are **10 bigmem nodes**, with **2TB of RAM** and **5.8TB local disk space**. 

Kestrel currently has no GPU nodes. They are expected to be available in Q2 of FY24. 


### Job Submission Recommendations:


## File System

Kestrel has a **95 PB** ClusterStor Lustre file system. Running jobs out of `/scratch` will be more performant than `/projects`. ScratchFS uses a Lustre file system in a hybrid flash-disk configuration providing a total of **27 petabytes** (PB) of capacity with **354 gigabytes (GB)/s** of IOR bandwidth. ProjectFS will provide **68 PB** of capacity with **200 GB/s** of IOR bandwidth. We advise running jobs out of `/scratch` and moving data to `/projects` for long term storage. Like on Eagle, `/scratch` will have a 28 day purge policy with no exceptions. 

The Home File System (HFS) on Kestrel is part of the ClusterStor used for PFS, providing highly reliable storage for user home directories and NREL-specific software. HFS will provide 1.2 PB of capacity. Snapshots of files on the HFS will be available up to 30 days after change/deletion. `/home` directories have a quota of 50 GB. 

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
* A recording of the Kestrel onboarding meeting, as well as the slides, are posted on the [Compuational Science Tutorials  team](https://teams.microsoft.com/l/team/19%3a6nLmPDt9QHQMEuLHVBaxfsitEZSGH6oXT6lyVauMvXY1%40thread.tacv2/conversations?groupId=22ad3c7b-a45a-4880-b8b4-b70b989f1344&tenantId=a0f29d7e-28cd-4f54-8442-7885aee7c080). 

## Contributions

The [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel) is open for contributions of examples, scripts, and other resources that would benefit the user community. To contribute, please open a Pull Request or contact [HPC-Help@nrel.gov](mailto://hpc-help@nrel.gov). To recommend topics to be covered, please open an [issue](https://github.com/NREL/HPC/issues) in the repository.

