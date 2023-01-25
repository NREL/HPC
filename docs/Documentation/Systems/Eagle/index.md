---
layout: default
title: File Systems 
has_children: true
hide:
 - toc
---

# File systems
Eagle has three primary file systems available for compute nodes. Understanding the usage of these is important for achieving the best performance. 

## NREL file systems
* **Home file system**
    * Quota of 50 GB
    * Used to hold scripts, source code, executables
* **Lustre parallel file system**: Accessiblle across all nodes. When using this file system please familiarize yourself with the [best practices section](Lustre/lustrebestpractices.md) 
    * /scratch/username
    * /projects
    * /shared-projects
    * /datasets
* **Node file system**: The local drive on each node, these are accessible only on a given node. 
    * /tmp/scratch
        * 1 TB HDD (spinning disk, average performance) on compute nodes with 196GB or less RAM
        * 1.6 TB SSD (higher performance) on 78 bigmem/GPU nodes
        * 25.6 TB SSD (higher performance, maximum local storage) on 20 bigmem/GPU "bigscratch" nodes
    
For more information on the file systems available on Eagle please see: [Eagle System Configuration](https://www.nrel.gov/hpc/eagle-system-configuration.html)
