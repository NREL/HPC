---
layout: default
title: Modules
parent: Vermillion
grand_parent: Systems
---

The page [Running](./running.md) describes running on Vermilion in more detail including a description of the hardware, partitions, simple build and run scripts and launching Vasp.

# Vermilion Modules and Applications
This page describes how to activate and use the modules available on Vermilion. Modules are not available by default on the machine.  This page discusses how to enable them.


## Selecting a user Environment 
Environments are provided with a number of commonly used modules including compilers, common build tools, optimized libraries, and some analysis tools. 

Since Vermilion is a new machine with an unusual architecture we are experimenting with environments. The environments are defined in date stamped subdirectories under the directory /nopt/nrel/apps.  Some of the environments in this directory are experimental and not intended for general use.   

User environments have a file myenv.\* in the date stamped directory.  These are for general use.  If a directory does not have a myenv.\* file then it is experimental, old, or not yet complete.  

The current user environments can be found by going to the directory /nopt/nrel/apps and looking for 
myenv.\* in sub directories.  For example

```
[joeuser2@vs-login-1 apps]$ ls -1 `pwd`/*/myenv*
/nopt/nrel/apps/210729a/myenv.2107300124
/nopt/nrel/apps/210901a/myenv.2109020548
/nopt/nrel/apps/210929a/myenv.2110041605
/nopt/nrel/apps/220525b/myenv.2110041605
[joeuser2@vs-login-1 apps]$ 
```

* 210729a
    * 	A bit dated but still should work.
* 210901a
    * 	A bit dated but still should work.
* 210929a
    * 	This is the recommended user environment.
* 220525b
    * 	Has some newer versions of compilers and other packages such as python 3.10.2 & gcc 12.1.



Currently, none of these environments are loaded by default for users.  Users must source one of the  /nopt/nrel/apps/210929a/myenv.\* files to enable an environment.  

The recommended environment is enabled by running the source command:


```
source /nopt/nrel/apps/210929a/myenv.2110041605
```

**NOTE:  You may want to add this line to your .bashrc file so modules are available at login.**

After sourcing this file you will have access to a set of modules. These can be listed using the following command:

```
module avail 
```

If you want to build applications you can then "module load" compilers and the like; for example

```
[joeuser2@vs-login-1 apps]$ ml gcc
[joeuser2@vs-login-1 apps]$ ml openmpi
```

will load gnu 9.4 and openmpi.  This will give you access to gcc, gfortran, mpicc, mpif90 and related commands.

You can load the Intel compilers (icc,icpc, ifort, mpiicc, mpiifort...) with the following commands.  Note you should also load gcc when using the Intel compilers because the Intel compilers actually use some gcc libraries.)

```
[joeuser2@vs-login-1 apps]$ ml intel-oneapi-compilers
[joeuser2@vs-login-1 apps]$ ml intel-oneapi-mpi
[joeuser2@vs-login-1 apps]$ ml gcc
[joeuser2@vs-login-1 apps]$ 
```

The python in this environment is very up to date, version 3.10.0.  It also contains many important packages including: numpy, scypi, matplotlib, pandas, jupyter, and jupyter-lab.

## Examples

The directory /nopt/nrel/apps/210929a/example contains some simple build and run scripts.  The directory /nopt/nrel/apps/210929a/example/vasp contains information about running Vasp.  These are discussed in more detail in the page [Running](./running.md).

