---
layout: default
title: Applications
parent: Vermillion
grand_parent: Systems
---

# Applications

The Vermilion HPC cluster marries traditional HPC deployments and modern cloud architectures, both using the OpenHPC infrastructure, and spack. [https://spack.io
](https://spack.io). 


There are a few packages installed using the OpenHPC infrastructure.  These can be found 
in */opt/ohpc/pub/*.  These are not in your path by default.  Some can be loaded via the module load command.  Running the command *module avail* you will see which of the packages can be loaded under the heading */opt/ohpc/pub/modulefiles*.  


However, there ary many additional modules that can be made available.  Instructions for enabling additional modules, Information about partitions, and running on Vermilion can be found in the documents
[Modules](./modules.md) and [Running](./running.md).

The page [Modules](./modules.md) discuses how to activate and use the modules on Vermilion. Modules are not available by default and must be activated.  Please see the [Modules](./modules.md) page for more information about setting up your environment and loading modules. 

The page [Running](./running.md) describes running on Vermilion in more detail including a description of the hardware, partitions, simple build and run scripts and launching Vasp.
