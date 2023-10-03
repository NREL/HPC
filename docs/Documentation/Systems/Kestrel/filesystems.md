---
layout: default
title: Filesystems
---
# Kestrel Filesystems


## Home File System

The Home File System (HFS) on Kestrel is part of the ClusterStor used for PFS, providing highly reliable storage for user home directories and NREL-specific software. HFS will provide 1.2 PB of capacity. Snapshots of files on the HFS will be available up to 30 days after change/deletion. 

**/home**

The /home directory on Kestrel is intended to hold small files. These include shell startup files, scripts, source code, executables, and data files. Each user has a quota of 50 GB.

**/nopt**

The /nopt directory on Kestrel resides on HFS and is where NREL-specific software, module files, licenses, and licensed software is kept.

## Parallel File System

The Parallel File System (PFS) ProjectFS and ScratchFS on Kestrel is a ClusterStor Lustre file system intended for high-performance I/O. ScratchFS uses a Lustre file system in a hybrid flash-disk configuration providing a total of 27 petabytes (PB) of capacity with 354 gigabytes (GB)/s of IOR bandwidth. ProjectFS will provide 68 PB of capacity with 200 GB/s of IOR bandwidth. 

**There are no backups of PFS data.**  Users are responsible for ensuring that critical data is copied to [Mass Storage](https://www.nrel.gov/hpc/mass-storage-system.html) or other alternate data storage location.

**/scratch**

Each user has their own directory in /scratch. Data in /scratch is subject to deletion after 28 days of inactivity. 

**/projects**

Each project/allocation has a directory in /projects intended to host data, configuration, and applications shared by the project.

**/shared-projects**

Projects may request a shared project directory to host data, configuration, and applications shared by multiple projects/allocations.

**/datasets**

The /datasets directory on Eagle hosts widely used data sets. 

There are multiple big data sets that are commonly used across various projects for computation and analysis on NREL's HPC Systems. We provide a common location on Eagle's scratch filesystem at /datasets, where these data sets are available for global reading by all compute nodes on Eagle. Each data set contains a readme file that covers background, references, explanation of the data structure, and Python examples.

**/datasets/NSRDB**

The National Solar Radiation Database (NSRDB) is a serially complete collection of meteorological and solar irradiance data sets for the United States and a growing list of international locations for 1998-2017. The NSRDB provides foundational information to support U.S. Department of Energy programs, research, and the general public.

**/datasets/WIND**

The Wind Integration National Data Set (WIND) Toolkit consists of wind resource data for North America and was produced using the [Weather Research and Forecasting Model (WRF)](https://www.mmm.ucar.edu/models/wrf).

## Node File System

Each Eagle compute node has a local solid-state drive (SSD) for use by compute jobs. They vary in size; 1 TB (standard), 1.6 TB (bigmem), and 25.6 TB (bigscratch), depending on the node feature requested. There are several possible scenarios in which a local disk may make your job run faster. For instance, you may have a job accessing or creating many small (temporary) files, you may have many parallel tasks accessing the same file, or your job may do many random reads/writes or memory mapping.

**/tmp/scratch**

The local disk is mounted at /tmp/scratch and set under the $LOCAL_SCRATCH environment variable during a job. A node will not have read or write access to any other node's local scratch, only its own. Also, this directory will be cleaned once the job ends. You will need to transfer any files to be saved to another file system. 

For more information about requesting this feature, please see Resource Request Descriptions on the [Eagle Batch Jobs](./Running/batch_jobs.md) page.

