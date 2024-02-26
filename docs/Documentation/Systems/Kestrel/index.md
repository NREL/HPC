---
layout: default
title: Kestrel
parent: Systems
has_children: true
hide:
 - toc
---

# About the Kestrel Cluster

Kestrel is configured to run compute-intensive and parallel computing jobs. It is a cluster comprised of 2454 nodes (servers) that run the Linux operating system (Red Hat Enterprise Linux), with a peak performance of 44 PetaFLOPS.

Please see the [System Configurations](../index.md) page for more information about hardware, storage, and networking.

!!! note
    GPUs are not currently available on Kestrel.  132 nodes with 4x Nvidia H100 GPUs are expected to be installed on Kestrel in FY24 Q2 (January, 2024).   

## Accessing Kestrel
Access to Kestrel requires an NREL HPC account and permission to join an existing allocation. Please see the [System Access](https://www.nrel.gov/hpc/system-access.html) page for more information on accounts and allocations.

#### For NREL Employees:

Users on an NREL device may connect via ssh to Kestrel from the NREL network using:

* kestrel.hpc.nrel.gov

 This will connect to one of the three login nodes using a round-robin load balancing approach. Users also have the option of connecting directly to an individual login node using one of the following names: 

* kl1.hpc.nrel.gov
* kl2.hpc.nrel.gov
* kl3.hpc.nrel.gov

#### For External Collaborators:
If you are an external HPC user, you will need a [One-Time Password Multifactor token (OTP)](https://www.nrel.gov/hpc/multifactor-tokens.html) for two-factor authentication.

For command line access, you may login directly to **kestrel.nrel.gov**.  Alternatively, you can connect to the [SSH gateway host](https://www.nrel.gov/hpc/ssh-gateway-connection.html).


## Data Analytics and Visualization (DAV) Nodes

There are eight DAV nodes available on Kestrel, which are nodes intended for HPC applications that require a graphical user interface.  They are not general-purpose remote desktops, and are intended for HPC or visualization software that requires Kestrel.

[FastX](https://nrel.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/) is available for HPC users to use graphical applications on the DAV nodes.

To connect to a DAV node using the load balancing algorithim, NREL employees can connect to **kestrel-dav.hpc.nrel.gov**. To connect from outside the NREL network, use **kestrel-dav.nrel.gov**. 


## Get Help With Kestrel
Please see the [Help and Support Page](../../help.md) for further information on how to seek assistance with Kestrel or your NREL HPC account. 
