---
layout: default
title: Singularity on Eagle
parent: Containers
---

Singularity is installed on Eagle's compute nodes as a module named `singularity-container`.  Images can be copied to eagle and run, or can be generated from a [recipe (definition file)](https://sylabs.io/guides/3.6/user-guide/definition_files.html). 

!!! note

    Input commands in the following examples are preceded by a `$`.

### Run hello-world ubuntu image

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
