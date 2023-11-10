---
layout: default
title: Filesystems
---
# Kestrel Filesystems


## Home File System

The Home File System (HFS) on Kestrel is part of the ClusterStor used for the Parallel File System (PFS), providing highly reliable storage for user home directories and NREL-specific software. HFS has 1.2 petabytes (PB) of capacity. Snapshots of files on the HFS are available up to 30 days after change/deletion. 

**/home**

The /home directory on Kestrel is intended to hold small files. These include shell startup files, scripts, source code, executables, and data files. Each user has a quota of 50 GB.

**/nopt**

The /nopt directory on Kestrel resides on HFS and is where NREL-specific software, module files, licenses, and licensed software are kept.

## Parallel File System

The Parallel File System (PFS) ProjectFS and ScratchFS on Kestrel is a ClusterStor Lustre file system intended for high-performance I/O. 

!!! warning 
    **There are no backups of PFS data.**  Users are responsible for ensuring that critical data is copied to [Mass Storage](../../Managing_Data/mss.md) or other alternate data storage location.

### ProjectFS

ProjectFS provides 68 PB of capacity with 200 GB/s of IOR bandwidth. It is intended for longer term data storage.

**/projects**

Each project/allocation has a directory in /projects intended to host data, configuration, and applications shared by the project.

Directories in /projects have a quota assigned based on the project resource allocation for that fiscal year. 

???+ info "To check your quota usage, run the following commands:"
    ```
    # To determine your Project ID run:

    [user@kl1 ~]$ lfs project -d /projects/csc000
    110255 P /projects/csc000

    # In this case, 110255 is the Project ID for project csc000.

    # To see usage towards your quota, run:

    [user@kl1 ~]$ lfs quota -hp 110255 /projects/csc000

    Disk quotas for prj 110255 (pid 110255):
        Filesystem    used   quota   limit   grace   files   quota   limit   grace 
    /projects/csc000    
                    617.5G    100T    100T       -  636875       0       0       -
    # An asterisk(*) by the used value indicates the project has exceeded its quota of storage, and writes to the directory are not allowed.
    ```

**/shared-projects**

Projects may request a shared project directory to host data, configuration, and applications shared by multiple projects/allocations. 

To request a /shared-projects directory, please contact [hpc-help@nrel.gov](mailto:HPC-Help@nrel.gov) and include the following information:
```
1. The name of the primary/"host" allocation that the /shared-projects directory will belong to. 
2. The name/email of a person who will authorize changes to the /shared-projects directory. 
3. How much space you would like to request (in TB). 
4. A list of other allocations that should have access to the /shared-projects directory. 
5. What you would like to call the directory. For example, "/shared-projects/myproject-shared" or other similar descriptive name, ideally between about 4-15 characters in length. 
6. A group name for the UNIX group ownership of the directory, the same or similar to the directory name provided in Step 5. 
```
!!! info
    If you currently have a shared project directory on Eagle that you need copied over to Kestrel, please contact [hpc-help@nrel.gov](mailto:HPC-Help@nrel.gov). 

### ScratchFS

ScratchFS is a Lustre file system in a hybrid flash-disk configuration providing a total of 27 petabytes (PB) of capacity with 354 gigabytes (GB)/s of IOR bandwidth. It is intended to support intensive I/O and we recommend running jobs out of ScratchFS for the best performance. 

**/scratch**

Each user has their own directory in /scratch. 

!!! warning 
    Data in /scratch is subject to deletion after 28 days of inactivity. It is recommended to store your important data, libraries, and programs on ProjectFS. 

### Datasets

We plan to have the datasets hosted on Eagle available on Kestrel in the near future. Please contact [hpc-help@nrel.gov](mailto:HPC-Help@nrel.gov) for current information about availability. 


## Node File System

Some Kestrel compute nodes have an NVMe local solid-state drive (SSD) for use by compute jobs. They vary in size; 1.7TB on 256 of the standard compute nodes and 5.8TB on the bigmem nodes. There are several possible scenarios in which a local disk may make your job run faster. For instance, you may have a job accessing or creating many small (temporary) files, you may have many parallel tasks accessing the same file, or your job may do many random reads/writes or memory mapping.

**/tmp/scratch**

The local disk is mounted at /tmp/scratch. A node will not have read or write access to any other node's local scratch, only its own. Also, this directory will be cleaned once the job ends. You will need to transfer any files to be saved to another file system. 

To request nodes with local disk, use the `--tmp` option in your job submission script. (e.g. `--tmp=1600000`). For more information about requesting this feature, please see the [Running on Kestrel page](./running.md).



