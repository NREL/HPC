## How to use singularity on eagle
Singularity is installed on CentOS 7 compute nodes as a module named singularity-container.  Images can be copied to eagle and run or can be generated from a [recipe](http://singularity.lbl.gov/docs-recipes). Input commands are preceded by a `$`

### Run hello-world ubuntu image

1. Log into compute node, checking it is running CentOS 7 

```
$ ssh el1.hpc.nrel.gov
[$USER@el1 ~]$ salloc -A MYALLOCATION -t 60 -N 1
[$USER@r1i3n18 ~]$ cat /etc/redhat-release 
CentOS Linux release 7.4.1708 (Core) 

```

2. Load the singularity-container module

```
[$USER@r1i3n18 ~]$ module purge
[$USER@r1i3n18 ~]$ module load singularity-container
```

3. Retrieve hello-world image.  Be sure to use /scratch as images are typically large

```
[$USER@r1i3n18 ~]$ cd /scratch/$USER
[$USER@r1i3n18 $USER]$ mkdir -p singularity-images
[$USER@r1i3n18 $USER]$ cd singularity-images
[$USER@r1i3n18 singularity-images]$ singularity pull --name hello-world.simg shub://vsoch/hello-world
Progress |===================================| 100.0% 
Done. Container is at: /lustre/eaglefs/scratch/$USER/singularity-images/hello-world.simg
```

4. Explore image details

```
[$USER@r1i3n18 singularity-images]$ singularity inspect hello-world.simg # Shows labels
{
    "org.label-schema.usage.singularity.deffile.bootstrap": "docker",
    "MAINTAINER": "vanessasaur",
    "org.label-schema.usage.singularity.deffile": "Singularity",
    "org.label-schema.schema-version": "1.0",
    "WHATAMI": "dinosaur",
    "org.label-schema.usage.singularity.deffile.from": "ubuntu:14.04",
    "org.label-schema.build-date": "2017-10-15T12:52:56+00:00",
    "org.label-schema.usage.singularity.version": "2.4-feature-squashbuild-secbuild.g780c84d",
    "org.label-schema.build-size": "333MB"
}
[$USER@r1i3n18 singularity-images]$ singularity inspect -r hello-world.simg # Shows the script run
#!/bin/sh 

exec /bin/bash /rawr.sh
```

5. Run image default script

```
[$USER@r1i3n18 singularity-images]$ singularity run hello-world.simg
RaawwWWWWWRRRR!!
```

6. Run in singularity bash shell

```
[$USER@r1i3n18 singularity-images]$ cat /etc/redhat-release 
CentOS Linux release 7.4.1708 (Core) 
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

## How to use singularity on peregrine
Singularity is installed on CentOS 7 compute nodes as a module named singularity-container.  Images can be copied to peregrine to run or can be generated from a [recipe](http://singularity.lbl.gov/docs-recipes).  In the examples below, output is only shown for test and run blocks.  Input commands are preceded by a `$`

### Run hello-world ubuntu image

1. Log into compute node, checking it is running CentOS 7 

```
ssh peregrine-login4.hpc.nrel.gov
qsub -I -A MYALLOCATION -q debug -l nodes=1,walltime=01:00:00
cat /etc/redhat-release 
```

2. Load the singularity-container module

```
module purge
module load singularity-container
```

3. Retrieve hello-world image.  Be sure to use /scratch as images are typically large

```
cd /scratch/$USER
mkdir -p singularity-images
cd singularity-images
singularity pull --name hello-world.simg shub://vsoch/hello-world
```

4. Explore image details

```
# Shows labels
$ singularity inspect hello-world.simg
{
    "org.label-schema.usage.singularity.deffile.bootstrap": "docker",
    "MAINTAINER": "vanessasaur",
    "org.label-schema.usage.singularity.deffile": "Singularity",
    "org.label-schema.schema-version": "1.0",
    "WHATAMI": "dinosaur",
    "org.label-schema.usage.singularity.deffile.from": "ubuntu:14.04",
    "org.label-schema.build-date": "2017-10-15T12:52:56+00:00",
    "org.label-schema.usage.singularity.version": "2.4-feature-squashbuild-secbuild.g780c84d",
    "org.label-schema.build-size": "333MB"
}
# Shows the script run
$ singularity inspect -r hello-world.simg
#!/bin/sh 

exec /bin/bash /rawr.sh
```

5. Run image default script

```
$ singularity run hello-world.simg
RaawwWWWWWRRRR!!
```

6. Run in singularity bash shell

```
# On compute node
$ cat /etc/redhat-release 
CentOS Linux release 7.3.1611 (Core) 
$ cat /etc/lsb-release 
cat: /etc/lsb-release: No such file or directory
# In singularity shell
$ singularity shell hello-world.simg
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

This example shows how to create a CentOS 7 singularity image with openmpi installed.  It requires root/admin privileges to create the image so must be run on a user's computer with singularity installed.  After creation, the image can be copied to peregrine and run.

1. Create a new recipe based on singularityhub/centos:latest

```
echo "Bootstrap: shub
From: singularityhub/centos:latest
" > centos-mpi.recipe
```

2. Install development tools and enable epel repository after bootstrap is created

```
echo "%post
  yum -y groupinstall "Development Tools"
  yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
" >> centos-mpi.recipe
```

3. Download, compile and install openmpi 2.1

```
echo "
curl -O https://download.open-mpi.org/release/open-mpi/v2.1/openmpi-2.1.2.tar.bz2
tar jxf openmpi-2.1.2.tar.bz2
cd openmpi-2.1.2
./configure --prefix=/usr/local
make
make install
" >> centos-mpi.recipe
```

4. Compile and install example mpi application

```
echo "
mpicc examples/ring_c.c -o ring
cp ring /usr/bin/
" >> centos-mpi.recipe

```

5. Install a package found in EPEL, in this example R

```
echo "  yum -y install R
" >> centos-mpi.recipe
```

6. Set default script to run ring

```
echo "%runscript
  /usr/bin/ring
" >> centos-mpi.recipe
```

7. Build image

```
sudo $(type -p singularity) build centos-mpi.simg centos-mpi.recipe
```

8. Test image

```
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
