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

### Create a CentOS 7 EPEL image with MPI support

This example shows how to create a CentOS 7 singularity image with openmpi installed.  Due to the need for admin priveleges, Singularity containers cannot be built on the HPC system.  Instead, the image can be built locally or using an online service, and then the image can be copied to Eagle and run.

!!! note

    Creating singularity images requires root/admin priveleges and cannot be done on the HPC systems.  Users that have access to a local Linux system with admin priveleges may run Singularity locally to build images.  Some options exist for running Singularity on Windows and Mac (see the [Singularity admin installation guide](https://docs.sylabs.io/guides/3.11/admin-guide/installation.html)), although support for Mac Silicone chips does not exist at the time of writing.  Another option is to use [Singularity Container Service](https://cloud.sylabs.io/), which provides free accounts with a limited amount of container build time.

**Step 1**: Create a new recipe based on singularityhub/centos:latest

```bash
echo "Bootstrap: shub
From: singularityhub/centos:latest
" > centos-mpi.recipe
```

**Step 2**: Install development tools and enable epel repository after bootstrap is created

```bash
echo "%post
  yum -y groupinstall "Development Tools"
  yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
" >> centos-mpi.recipe
```

**Step 3**: Download, compile and install openmpi 2.1

```bash
echo "
curl -O https://download.open-mpi.org/release/open-mpi/v2.1/openmpi-2.1.2.tar.bz2
tar jxf openmpi-2.1.2.tar.bz2
cd openmpi-2.1.2
./configure --prefix=/usr/local
make
make install
" >> centos-mpi.recipe
```

**Step 4**: Compile and install example mpi application

```bash
echo "
mpicc examples/ring_c.c -o ring
cp ring /usr/bin/
" >> centos-mpi.recipe
```

**Step 5**: Install a package found in EPEL, in this example R

```bash
echo "  yum -y install R
" >> centos-mpi.recipe
```

**Step 6**: Set default script to run ring

```bash
echo "%runscript
  /usr/bin/ring
" >> centos-mpi.recipe
```

**Step 7**: Build image

```bash
sudo $(type -p singularity) build centos-mpi.simg centos-mpi.recipe
```

**Step 8**: Test image

```bash
$ mpirun -np 20 singularity exec centos-mpi.simg /usr/bin/ring
$ singularity run centos-epel-r.simg --version
R version 3.4.4 (2018-03-15) -- "Someone to Lean On"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-redhat-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under the terms of the
GNU General Public License versions 2 or 3.
For more information about these matters see
http://www.gnu.org/licenses/.

$ singularity exec centos-mpi.simg Rscript -e "a <- 42; A <- a*2; print(A)"
[1] 84
```
