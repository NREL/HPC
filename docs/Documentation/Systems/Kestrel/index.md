---
layout: default
title: Kestrel
parent: Systems
has_children: true
hide:
 - toc
---

# About the Kestrel Cluster

Kestrel is configured to run compute-intensive and parallel computing jobs. It is a heterogeneous system comprised of 2,314 CPU-only nodes, and 156 GPU-accelerated nodes that run the Linux operating system (Red Hat Enterprise Linux), with a peak performance of 44 PetaFLOPS.

Please see the [System Configurations](../index.md) page for more information about hardware, storage, and networking.

## Accessing Kestrel
Access to Kestrel requires an NREL HPC account and permission to join an existing allocation. Please see the [System Access](https://www.nrel.gov/hpc/system-access.html) page for more information on accounts and allocations.

Kestrel has two types of login nodes, CPU and GPU, which share the same architecture as the corresponding compute nodes. You should use the CPU login nodes to compile software for use on and to submit jobs to the CPU compute nodes, and the GPU login nodes for GPU jobs.  


#### For NREL Employees:

Users on an NREL device may connect via ssh to Kestrel from the NREL network using:

* kestrel.hpc.nrel.gov (CPU)
* kestrel-gpu.hpc.nrel.gov (GPU)

 This will connect to one of the three login nodes using a round-robin load balancing approach. Users also have the option of connecting directly to an individual login node using one of the following names: 

* kl1.hpc.nrel.gov (CPU)
* kl2.hpc.nrel.gov (CPU)
* kl3.hpc.nrel.gov (CPU)
* kl5.hpc.nrel.gov (GPU)
* kl6.hpc.nrel.gov (GPU)

#### For External Collaborators:
If you are an external HPC user, you will need a [One-Time Password Multifactor token (OTP)](https://www.nrel.gov/hpc/multifactor-tokens.html) for two-factor authentication.

For command line access, you may login directly to **kestrel.nrel.gov**.  Alternatively, you can connect to the [SSH gateway host](https://www.nrel.gov/hpc/ssh-gateway-connection.html) or the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html).

To access the GPU login nodes, first connect with one of the methods described above, and then ssh to **kestrel-gpu.hpc.nrel.gov**. 

!!! warning "Login Node Policies"
    Kestrel login nodes are shared resources, and because of that are subject to process limiting based on usage to ensure that these resources aren't being [used inappropriately](https://www.nrel.gov/hpc/inappropriate-use-policy.html). Each user is permitted up to 8 cores and 100GB of RAM at a time, after which the Arbiter monitoring software will begin moderating resource consumption, restricting further processes by the user until usage is reduced to acceptable limits.

## Data Analytics and Visualization (DAV) Nodes

There are eight DAV nodes available on Kestrel, which are nodes intended for HPC applications that require a graphical user interface.  They are not general-purpose remote desktops, and are intended for HPC or visualization software that requires Kestrel.

[FastX](https://nrel.github.io/HPC/Documentation/Viz_Analytics/virtualgl_fastx/) is available for HPC users to use graphical applications on the DAV nodes.

To connect to a DAV node using the load balancing algorithim, NREL employees can connect to **kestrel-dav.hpc.nrel.gov**. To connect from outside the NREL network, use **kestrel-dav.nrel.gov**. 


## Get Help With Kestrel
Please see the [Help and Support Page](../../help.md) for further information on how to seek assistance with Kestrel or your NREL HPC account. 
