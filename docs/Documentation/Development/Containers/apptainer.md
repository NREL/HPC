## How to use Apptainer (Singularity) on Kestrel

Singularity has been deprecated in favor of a new container runtime environment called Apptainer, which is its direct decendent. Apptainer will run Singularity containers and it supports Singularity commands by default. On Kestrel, `singularity` is an alias for `apptainer` and the two commands can be used interchangeably in most instances. However, since Singularity is deprecated, it is advised to use Apptainer.

This page describes 

More information about Apptainer can be found at [https://apptainer.org](https://apptainer.org). 

On Kestrel, Apptainer is installed on compute nodes and is accessed via a module named `apptainer` (default: `apptainer/1.1.9`). The directory `/nopt/nrel/apps/software/apptainer/1.1.9/examples` holds a number of images (`*.sif`) and an example script (`script`) that shows how to run containers hosting MPI programs across multiple nodes.  

Before we get to the more complicated example from `script`, we'll first look at downloading (or *pulling*) and working with a simple image.

Input commands are preceded by a `$`.

!!! note
    If you wish to containerize your own application, it may be worth starting with [building a local Docker image and transferring it to Kestrel](./index.md#example-docker-build-workflow-for-hpc-users) before attempting to directly create your own Apptainer image, since you do not have root access on Kestrel. The following 

## Apptainer runtime examples

### Run hello-world Ubuntu image

##### Allocate a compute node.

```
$ ssh USERNAME@kestrel.hpc.nrel.gov
[USERNAME@kl1 ~]$ salloc --exclusive --mem=0 --tasks-per-node=104 --nodes=1 --time=01:00:00 --account=MYACCOUNT --partition=debug
[USERNAME@x1000c0s0b0n0 ~]$ cat /etc/redhat-release
Red Hat Enterprise Linux release 8.6 (Ootpa)

```

##### Load the apptainer module

```
[USERNAME@x1000c0s0b0n0 ~]$ module purge
[USERNAME@x1000c0s0b0n0 ~]$ module load apptainer/1.1.9
```

##### Retrieve hello-world image.  Be sure to use /scratch as images are typically large

```
[USERNAME@x1000c0s0b0n0 ~]$ cd /scratch/$USER
[USERNAME@x1000c0s0b0n0 USERNAME]$ mkdir -p apptainer-images
[USERNAME@x1000c0s0b0n0 USERNAME]$ cd apptainer-images
[USERNAME@x1000c0s0b0n0 apptainer-images]$ apptainer pull --name hello-world.simg shub://vsoch/hello-world
Progress |===================================| 100.0%
```

##### Explore image details

```
[USERNAME@x1000c0s0b0n0 apptainer-images]$ apptainer inspect hello-world.simg # Shows labels
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
[USERNAME@x1000c0s0b0n0 apptainer-images]$ apptainer inspect -r hello-world.simg # Shows the script run
#!/bin/sh

exec /bin/bash /rawr.sh
```

##### Run image default script

```
[USERNAME@x1000c0s0b0n0 apptainer-images]$ apptainer run hello-world.simg
RaawwWWWWWRRRR!! Avocado!
```

### Run images containing MPI programs on multiple nodes



As mentioned above, there is a script in the apptainer directory that shows how MPI applications built inside an image can be run on multiple nodes. We'll run 5 containers with different versions of MPI. Each container has two MPI programs installed, a glorified Hello World (`phostone`) and PingPong (`ppong`). The 5 versions of MPI are:

1. openmpi
1. IntelMPI
1. MPICH - with ch4
1. MPICH - with ch4 with different compile options
1. MPICH - with ch3

"ch*" can be thought as a "lower level" communications protocol. A MPICH container might be built with either but we have found that ch4 is considerably faster on Kestrel. 

The script can be found at `/nopt/nrel/apps/software/apptainer/1.1.9/examples/script`, as well as [our GitHub repository](https://github.com/NREL/HPC/blob/master/kestrel/apptainer/script).

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

We set the variable `CDIR` which points to the directory from which we will get our containers.

We next create a directory for our run and go there. The `cat` and `printenv` commands give us a copy of our script and the environment in which we are running. This is useful for debugging.

Before we run the MPI containers, we run the command `hostname` from inside a very simple container `alpine.sif`. We show containers can be run without/with `srun`. In the second instance we `cat /etc/os-release` to show we are running a different OS.  

Then we get into the MPI containers. This is done in a loop over containers containing the MPI versions: `openmpi`, `intelmpi`, `mpich_ch4`, `mpich_ch4b`, and `mpich_ch3`. 

The application `tymer` is a simple wall clock timer.  

The `--mpi=` option on the srun line instructs slurm how to launch jobs. The normal option is `--mpi=pmi2`. However, containers using OpenMPI might need to use the option `--mpi=pmix` as we do here.

The first loop just runs a quick "hello world" example. The second loop runs a pingpong test. We skip the `mpich_ch3` pingpong test because it runs very slowly.

You can see example output from this script in the directory:

```
/nopt/nrel/apps/software/apptainer/1.1.9/examples/output/
```

Within `/nopt/nrel/apps/software/apptainer/1.1.9/examples`, the subdirectory `defs` contains the recipes for the images in `examples`. The images `apptainer.sif` and `intel.sif` were built in two steps using `app_base.def` - apptainer.def and mods_intel.def - intel.def. They can also be found in the [HPC code examples repository](https://github.com/NREL/HPC/tree/master/kestrel/apptainer/defs).

The script `sif2def` can be used to generate a `.def` recipe from a `.sif` image. It has not been extensively tested, so it may not work for all images and is provided here "as is."

## Apptainer buildtime examples

### Create Ubuntu-based image with MPI support

Apptainer images can be generated from a `.def` [recipe](https://apptainer.org/docs/user/main/build_a_container.html). 

This example shows how to create an Apptainer image running on the Ubuntu operating system with openmpi installed. The recipe is shown in pieces to make it easier to describe what each section does. The complete recipe can be found in the `defs` subdirectory of `/nopt/nrel/apps/software/apptainer/1.1.9/examples`. Building images requires root/admin privileges, so the build process must be run on a user's computer with apptainer installed or via the [Singularity Container Service](https://cloud.sylabs.io/). After creation, the image can be [copied to Kestrel](../../Managing_Data/Transferring_Files/index.md) and run.

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

##### Compile and install example MPI application

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
[kuser@kl1 ~]$ salloc --exclusive --mem=0 --tasks-per-node=104 --nodes=2 --time=01:00:00 --account=MYACCOUNT --partition=debug
salloc: Granted job allocation 90367
salloc: Waiting for resource configuration
salloc: Nodes x3000c0s25b0n0,x3000c0s27b0n0 are ready for job
[kuser@x3000c0s25b0n0 ~]$ module load apptainer 
[kuser@x3000c0s25b0n0 ~]$ srun -n 8 --tasks-per-node=4 --mpi=pmix apptainer run small.sif
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

## Utilizing GPU resources with Apptainer images

GPU-accelerated software often have complex software and hardware requirements to function properly, making containerization a particularly attractive option for deployment and use. These requirements manifest themselves as you are building your image (*buildtime*) and when you run a container (*runtime*). This section describes key components of software images that are successfully GPU-enabled. For more detailed documentation on the subject, visit Apptainer's dedicated [GPU Support page](https://apptainer.org/docs/user/1.0/gpu.html).

#### Considerations during buildtime

##### GPU driver and CUDA compatibility

A major consideration you should keep in mind when pulling or building GPU-enabled images is the version of both the GPU drivers and [CUDA](https://developer.nvidia.com/cuda-toolkit) on the host system. You can obtain this information from the host by running the command `nvidia-smi` after allocating a GPU within a Slurm job on your desired system. Alternatively, you could simply consult the table below. Note that the GPU drivers/CUDA versions depend on a given GPU partition. If your running container is installed with a different GPU driver/CUDA version than what is listed below for your target system, you will either run into a fatal error, or the software will bypass the GPU and run on the CPU, slowing computation. For example, an [image containing GPU-accelerated TensorFlow](../../Machine_Learning/Containerized_TensorFlow/index.md) might work perfectly fine on Eagle, but its drivers would likely be too old for either of Kestrel's GPU partitions.

| System   | Partition name | GPU type<br>(cards per node) | `nvidia-smi`<br>GPU driver version | CUDA Version |
|:--------:|:--------------:|:----------------------------:|:----------------------------------:|:------------:|
| Kestrel  | gpu-h100       | H100 (4)                     | 545.23.08                          | 12.3         |
| Swift    | gpu            | A100 (4)                     | 545.23.08                          | 12.3         |
| Eagle    | gpu            | V100 (2)                     | 460.32.03                          | 11.2         |
| Vermilion| gpu            | A100 (1)                     | 460.106.00                         | 11.2         |

#### Considerations during runtime

##### Allocate a Slurm job with a GPU resource

As with running any GPU-enabled software, you first want to ensure that you have the expected hardware available to your computing environment. On NREL's HPC systems, this means [submitting a Slurm job](../../Slurm/index.md) that requests at least one GPU card. You will need to submit the job to a GPU partition and explicitly request how many GPU cards (per node) you wish to run on with the `--gres=gpu:<number>` option. As an example, this uses `salloc` to request an interactive job that allocates one NVIDIA H100 GPU card (`--gres=gpu:1`) on one node (`-N 1`) on Kestrel:

```
salloc -A <account> -p gpu-h100 -t 01:00:00 -N 1 --gres=gpu:1
```

You can verify whether the GPU device is found by running `nvidia-smi` after landing on the node. If you receive any kind of output other than a `No devices were found` message, there is a GPU waiting to be used by your software.

!!! note
    If you submit to the `gpu-h100` partition without including `--gres=gpu:<# of GPUs per node>`, your job will *not* allocate any GPU cards to your session, preventing you from using the resource at all.

##### Provide the `--nv` flag to Apptainer runtime

Once you allocate at least one GPU card in your job, you then need to make Apptainer recognize the GPU resources you wish to use. To accomplish this, you can supply the `--nv` flag to the `apptainer shell ...` or `apptainer exec ...` command. Using a generic `gpu_accelerated_tensorflow.sif` image as an example:

```
apptainer exec --nv gpu_accelerated_tensorflow.sif tensorflow.py
```

## Best practices and recommendations

This section describes general recommendations and best practices for Apptainer users across NREL's HPC systems.

##### Change Apptainer cache location to `$LOCAL_SCRATCH`

By default, Apptainer will cache image layers to your `$HOME` folder when you pull or build `.sif` images, which is not ideal as users have a limited storage quota in /home. As you continue to use Apptainer, this cache folder can become quite large and can easily fill your `$HOME`. Fortunately, the location of this cache folder can be controlled through the `APPTAINER_CACHEDIR` environmental variable. To avoid overfilling your $HOME with unnecessary cached data, it is recommended to add an `APPTAINER_CACHEDIR` location to your `~/.bashrc` file. You can accomplish this with the following command, which will direct these layers to save to a given system's scratch space:

`echo "export APPTAINER_CACHEDIR=$LOCAL_SCRATCH/.apptainer" >> ~/.bashrc`

Note that you will either need to log out and back into the system, or run `source ~/.bashrc` for the above change to take effect.

!!! note
    Swift users will need to request that a home directory is created for them by emailing [hpc-help@nrel.gov](mailto:hpc-help@nrel.gov).

##### Save `.def` files to home folder and images to `$LOCAL_SCRATCH` or /projects

An Apptainer definition file (`.def`) is a relatively small text file that contains much (if not all) of the build context for a given image. Since your `$HOME` folders on NREL's HPC systems are regularly backed up, it is strongly recommended to save this file to your home directory in case it accidentally gets deleted or otherwise lost. Since `.sif` images themselves are 1. typically large and 2. can be rebuilt from the `.def` files, we recommend saving them to a folder outside of your `$HOME`, for similar reasons described in the previous section. If you intend to work with an image briefly or intermittantly, it may make sense to save the `.sif` to your `$LOCAL_SCRATCH` folder, from which files get automatically deleted after 30 days. If you plan to use an image frequently over time or share it with other users in your allocation, saving it in a `/projects/` location you have access to may be better.




