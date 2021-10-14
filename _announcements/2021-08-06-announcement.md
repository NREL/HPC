---
title: August 2021 NREL HPC Monthly Update
data: 2021-08-06
layout: default
brief: CSC User & Applications Support, Eagle Compute Node Local Storage
---

# August CSC User & Applications Support
Q-Chem 5.4 will be made generally available. Look for the q-chem/5.4 module to appear in module avail output.


# Eagle Compute Node Local Storage and Limitations in /tmp
The compute nodes on Eagle have local disks that may benefit your workload. Disk sizes 
start at 1TB, with a limited number of nodes available with more storage. The storage is available at /tmp/scratch. 

/tmp is in memory, and is limited in size. We occasionally see jobs filling /tmp, and recommend users adjust their 
environment to use /tmp/scratch.


