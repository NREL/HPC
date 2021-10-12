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
* **Lustre parallel file system**: Accessiblle across all nodes. When using this file system please familiarize yourself with the [best practices section](/Documentation/Data-and-File-Systems/File-Systems/Lustre/lustrebestpractices) 
    * /scratch/username
    * /projects
    * /shared-projects
    * /datasets
* **Node file system**: The local drive on each node, these are accessible only on a given node. 
    * /tmp/scratch

For more information on the file systems available on Eagle please see: [Eagle System Configuration](https://www.nrel.gov/hpc/eagle-system-configuration.html)
