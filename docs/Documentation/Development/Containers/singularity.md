---
layout: default
title: Singularity
parent: Containers
---

As discussed in [Intro to Containers](index.md), Singularity is a platform designed specifically for running containers on HPC systems. Images can be built locally and copied to the HPC system or pulled from an online registry.  For more information about building containers, see [here](index.md#building).

The table below shows the appropriate commands for loading Singularity on each system:

| System     | Module command                      |
|------------|-------------------------------------|
| Eagle      | `module load singularity-container` |
| Swift      | `module load singularity`           |
| Vermilion  | `module load singularity`           |
| Kestrel    | `module load apptainer`             | 

!!! note
    Singularity has been deprecated in favor of a new container application called Apptainer. For more information about Apptainer and using it on Kestrel, see [Apptainer](./apptainer.md).

### Run hello-world ubuntu image on Eagle

The following example shows how to download and run a simple "hello-world" container based on Ubuntu.  The example is written for Eagle but can be adapated to other systems by using the appropriate module command.

!!! note

    Input commands in the following examples are preceded by a `$`.

**Step 1**: Log into compute node, checking it is running CentOS 7 

```bash
$ ssh eagle.hpc.nrel.gov
[$USER@el1 ~]$ srun -A MYALLOCATION -t 60 -N 1 --pty $SHELL
[$USER@r1i3n18 ~]$ cat /etc/redhat-release 
CentOS Linux release 7.7.1908 (Core) 
```

**Step 2**: Load the `singularity-container` module

```bash
[$USER@r1i3n18 ~]$ module purge
[$USER@r1i3n18 ~]$ module load singularity-container
```

**Step 3**: Retrieve `hello-world` image.  Be sure to use `/scratch`, as images are typically large

```bash
[$USER@r1i3n18 ~]$ cd /scratch/$USER
[$USER@r1i3n18 $USER]$ mkdir -p singularity-images
[$USER@r1i3n18 $USER]$ cd singularity-images
[$USER@r1i3n18 singularity-images]$ singularity pull --name hello-world.simg shub://vsoch/hello-world
Progress |===================================| 100.0% 
Done. Container is at: /lustre/eaglefs/scratch/$USER/singularity-images/hello-world.simg
```

**Step 4**: Run image default script

```bash
[$USER@r1i3n18 singularity-images]$ singularity run hello-world.simg
RaawwWWWWWRRRR!! Avocado.
```

!!! note

    Running the image may produces errors such as:
    
    ```
    ERROR: ld.so: object '/nopt/xalt/xalt/lib64/libxalt_init.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
    ```
    
    This can be resolved by unsetting `LD_PRELOAD`:
    
    ```bash
    $ unset LD_PRELOAD
    ```

**Step 5**: Run in singularity bash shell

```bash
[$USER@r1i3n18 singularity-images]$ cat /etc/redhat-release 
CentOS Linux release 7.7.1908 (Core)
[$USER@r1i3n18 singularity-images]$ cat /etc/lsb-release 
cat: /etc/lsb-release: No such file or directory

[$USER@r1i3n18 singularity-images]$ singularity shell hello-world.simg
Singularity: Invoking an interactive shell within container...

Singularity hello-world.simg:~> cat /etc/lsb-release 
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=14.04
DISTRIB_CODENAME=trusty
DISTRIB_DESCRIPTION="Ubuntu 14.04.5 LTS"
Singularity hello-world.simg:~> cat /etc/redhat-release 
cat: /etc/redhat-release: No such file or directory
```
