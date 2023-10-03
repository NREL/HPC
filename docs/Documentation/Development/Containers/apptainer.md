## How to use Apptainer (Singularity) on Kestrel

Singularity has been deprecated in favor of a new container application called apptainer. Apptainer is a decendent of singularity.  Apptainer will run singularity  containers and it supports singularity commands. On Kestrel singularity is an alias for apptainer and the two commands can be use interchangeably in most instances. However, since singularity is depricated it is advised to use apptainer.

More information about apptainer can be found at [https://apptainer.org](https://apptainer.org). 

Apptainer is installed on compute nodes and is accessed via a module named *apptainer*.  

The directory /nopt/nrel/apps/software/apptainer/1.1.9/examples
holds a number of containers and an example script that shows how to run containers hosting MPI programs across multiple nodes.  

Before we get to more complicated examples we'll first look at downloading and working with a simple remote image.

Input commands are preceded by a `$`

### Run hello-world ubuntu image

##### Log into compute node.

```
$ ssh kl1.hpc.nrel.gov
[$kuser@el1 ~]$ salloc --exclusive --mem=0 --tasks-per-node=104 --nodes=1 --time=01:00:00 --account=MYACCOUNT --partition=debug
[$kuser@r1i3n18 ~]$ cat /etc/redhat-release
Red Hat Enterprise Linux release 8.6 (Ootpa

```

##### Load the apptainer module

```
[$kuser@r1i3n18 ~]$ module purge
[$kuser@r1i3n18 ~]$ module load apptainer
```

##### Retrieve hello-world image.  Be sure to use /scratch as images are typically large

```
[$kuser@r1i3n18 ~]$ cd /scratch/$USER
[$kuser@r1i3n18 $kuser]$ mkdir -p apptainer-images
[$kuser@r1i3n18 $kuser]$ cd apptainer-images
[$kuser@r1i3n18 apptainer-images]$ apptainer pull --name hello-world.simg shub://vsoch/hello-world
Progress |===================================| 100.0%
Done. Container is at: /lustre/eaglefs/scratch/$USER/apptainer-images/hello-world.simg
```

##### Explore image details

```
[$kuser@r1i3n18 apptainer-images]$ apptainer inspect hello-world.simg # Shows labels
{
    "org.label-schema.usage.apptainer.deffile.bootstrap": "docker",
    "MAINTAINER": "vanessasaur",
    "org.label-schema.usage.apptainer.deffile": "apptainer",
    "org.label-schema.schema-version": "1.0",
    "WHATAMI": "dinosaur",
    "org.label-schema.usage.apptainer.deffile.from": "ubuntu:14.04",
    "org.label-schema.build-date": "2017-10-15T12:52:56+00:00",
    "org.label-schema.usage.apptainer.version": "2.4-feature-squashbuild-secbuild.g780c84d",
    "org.label-schema.build-size": "333MB"
}
[$kuser@r1i3n18 apptainer-images]$ apptainer inspect -r hello-world.simg # Shows the script run
#!/bin/sh

exec /bin/bash /rawr.sh
```

##### Run image default script

```
[$kuser@r1i3n18 apptainer-images]$ apptainer run hello-world.simg
RaawwWWWWWRRRR!! Avocado!

### Run images containing MPI programs on multiple nodes

```

As mentioned above there is a script in the apptainer directory that shows how MPI applications built inside a container image can be run on multiple nodes. We'll look at 5 containers with different versions of MPI. Each container has two MPI programs installed, a glorified Hello World (phostone) and PingPong (ppong).  The 5 versions of MPI are

1. openmpi
1. IntemMPI
1. MPICH - with ch4
1. MPICH - with ch4 with different compile options
1. MPICH - with ch3

"ch*" can be thought as a "lower level" communications protocol.  A MPICH container might be build with either but we have found that ch4 is considerably faster on Kestrel. 

The script can be found at /nopt/nrel/apps/software/apptainer/1.1.9/examples/script and at [https://github.com/NREL/HPC/tree/master/kestrel/apptainer](https://github.com/NREL/HPC/tree/master/kestrel/apptainer)

Here is a copy:

```

#!/bin/bash 
#SBATCH --job-name="apptainer"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --output=apptainer.log
#SBATCH --mem=0

export STARTDIR=`pwd`
export CDIR=/nopt/nrel/apps/software/apptainer/1.1.9/examples
mkdir $SLURM_JOB_ID
cd $SLURM_JOB_ID

cat $0 >   script
printenv > env

touch warnings
touch output

module load apptainer
which apptainer >> output

echo "hostname" >> output
hostname        >> output

echo "from alpine.sif" >> output
          apptainer exec $CDIR/alpine.sif hostname  >> output
echo "from alpine.sif with srun" >> output
srun -n 1 --nodes=1 apptainer exec $CDIR/alpine.sif cat /etc/os-release  >> output


export OMP_NUM_THREADS=2

$CDIR/tymer times starting

MPI=pmix
for v in openmpi intel mpich_ch4 mpich_ch4b  mpich_ch3; do
  srun  --mpi=$MPI   apptainer  exec   $CDIR/$v.sif  /opt/examples/affinity/tds/phostone -F >  phost.$v  2>>warnings
  $CDIR/tymer times $v
  MPI=pmi2
  unset PMIX_MCA_gds
done

MPI=pmix
#skip mpich_ch3 because it is very slow
for v in openmpi intel mpich_ch4 mpich_ch4b           ; do
  srun  --mpi=$MPI   apptainer  exec   $CDIR/$v.sif  /opt/examples/affinity/tds/ppong>  ppong.$v  2>>warnings
  $CDIR/tymer times $v
  MPI=pmi2
  unset PMIX_MCA_gds
done

$CDIR/tymer times finished

mv $STARTDIR/apptainer.log .
         
```

We set the variable CDIR which points to the directory from which we will get our containers.

We next create a dirctory for our run and go there. The `cat` and `printenv`commands give us a copy of our script and the environment in which we are running. This is useful for debugging.


Before we run the MPI containers we run the command `hostname` from inside a very simple container `alpine.sif`.  We show containers can be run without/with `srun`. In the second instance we `cat /etc/os-release` to show we are running a different OS.  

Then we get into the MPI containers. This is done in a loop over containers containing the MPI versions: openmpi, intelmpi, mpich_ch4, mpich_ch4b, and mpich_ch3. 

The application *tymer* is a simple wall clock timer.  

The *--mpi=* option on the srun line instructs slurm how to launch jobs. The normal option is `--mpi=pmi2`.  However, containers using OpenMPI might need to use the option `--mpi=pmix` as we do here.

The first loop just runs a quick "hello world" example. The second loop runs a pingpong test. We skip mpich_ch3 pingpong test because it runs very slowly.

You can see example output from this script in the directory:

```
/nopt/nrel/apps/software/apptainer/1.1.9/examples/output/
```

The directory /nopt/nrel/apps/software/apptainer/1.1.9/examples/defs containes the recipes for the containers. The containers `apptainer.sif`` and `intel.sif` were built in two steps using app_base.def - apptainer.def and mods_intel.def - intel.def. They can also be found at [https://github.com/NREL/HPC/tree/master/kestrel/apptainer](https://github.com/NREL/HPC/tree/master/kestrel/apptainer)

The script `sif2def` can be used to generate a recipe from a container. It has not been extensively tested and may not work for all containers and is provided here "as is."


### Create a Ubuntu image with MPI support

Images can be generated from a [recipe](https://apptainer.org/docs/user/main/build_a_container.html). 

This example shows how to create a Ubuntu singularity image with openmpi installed. The recipe is shown in pieces to make it easier to describe what each section does.  The complete recipe can be found in the `defs` directory. Building containers normally requires root/admin privileges so the build process must be run on a user's computer with apptainer installed.  After creation, the image can be copied to Kestrel and run.  

##### Create a new recipe based on ubuntu:latest

```
Bootstrap: docker
from: ubuntu:latest

```
##### Add LD\_LIBRARY\_PATH /usr/local/lib used by OpenMPI

```
%environment
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    export PMIX_MCA_gds=^ds12
```

##### Install development tools after bootstrap is created

```
%post
    echo "Installing basic development packages..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y bash gcc g++ gfortran make curl python3

```

##### Download, compile and install openmpi. 
```
    echo "Installing OPENMPI..."
    curl https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz --output openmpi-4.1.5.tar.gz
    mkdir -p /opt/openmpi/src
    tar -xzf openmpi-4.1.5.tar.gz -C /opt/openmpi/src
    cd /opt/openmpi/src/*
    ./configure 
    make install
```

##### Compile and install example mpi application

```
    echo "Build OPENMPI example..."
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    cd /opt/openmpi/src/*/examples
    mpicc ring_c.c -o /usr/bin/ring

```


##### Set default script to run ring

```
  /usr/bin/ring
```

##### Example Build image command (must have root access)

```
sudo $(type -p apptainer) build small.sif  ubuntu-mpi.def
```

##### Test image

```
[kuser@kl1.hpc.nrel.gov ~]$salloc --exclusive --mem=0 --tasks-per-node=104 --nodes=2 --time=01:00:00 --account=MYACCOUNT --partition=debug
salloc: Granted job allocation 90367
salloc: Waiting for resource configuration
salloc: Nodes x3000c0s25b0n0,x3000c0s27b0n0 are ready for job
[kuser@x3000c0s25b0n0 ~]$module load apptainer 
[kuser@x3000c0s25b0n0 ~]$srun -n 8 --tasks-per-node=4 --mpi=pmix apptainer run small.sif
Process 2 exiting
Process 3 exiting
Process 0 sending 10 to 1, tag 201 (8 processes in ring)
Process 0 sent to 1
Process 0 decremented value: 9
Process 0 decremented value: 8
Process 0 decremented value: 7
Process 0 decremented value: 6
Process 0 decremented value: 5
Process 0 decremented value: 4
Process 0 decremented value: 3
Process 0 decremented value: 2
Process 0 decremented value: 1
Process 0 decremented value: 0
Process 0 exiting
Process 1 exiting
Process 5 exiting
Process 6 exiting
Process 7 exiting
Process 4 exiting
[kuser@x3000c0s25b0n0 ~]$

```








