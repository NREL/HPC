---
layout: default
title: Containers Intro
parent: Containers
order: 1
---

# Introduction to Software Containerization

## What are software images/containers?
Software *images* provide a method of packaging your code so that its *container* can be run anywhere you have a *container runtime environment*. This enables you to create an image on your local laptop and then run it on an HPC system or other computing resource. Software containerization provides an alternative, more robust method of isolating and packaging your code compared to solutions such as Conda virtual environments. 

**A note on terminology**: A software *container* is considered an instance of an *image*, meaning the former gets created during the runtime of the latter. In other words, a software *image* is what you build and distribute, whereas the *container* is what gets executed from a given image.


## Docker vs. Apptainer
The most common container runtime environment (outside of HPC) is Docker. Due to the fact that it requires root-level permissions to build its associated images and run containers, Docker is not suited for HPC environments and is therefore not available on NREL's systems currently. Apptainer is an alternative containerization tool that can be used in HPC environments because running it does not require root. However, you can use Docker to build images locally and convert them to the Apptainer format for use with HPC (described in more detail [here](index.md#building-software-images)).

## Compatibility 
Apptainer is able to run most Docker images, but Docker is unable to run Apptainer images. A key consideration when deciding to containerize an application is which container engine to build with. A suggested best practice is to build images with Docker whenever possible, as this provides more flexibility. However, if this is not feasible, you may have to build with Apptainer or maintain separate images for each container engine.

## Advantages to software containerization
* **Portability**: Containers can be run on HPC, locally, and on cloud infrastructure used at NREL. 
* **Reproducibility**: Containers are one option to ensure reproducible research by packaging all necessary software to reproduce an analysis. Containers are also easily versioned using a hash.
* **Modularity**: Images are composed of cacheable "layers" of other images or build commands, facilitating the image building process.
* **Workflow integration**: Workflow management systems such as Airflow, Nextflow, Luigi, and others provide built-in integration with container engines. 

## Accessing HPC hardware from software containers
Both Apptainer and Docker provide the ability to use hardware based features on the HPC systems such as GPUs. A common usage of containers is packaging GPU-enabled tools such as TensorFlow. Apptainer natively provides access to the GPU and driver on the host. Please visit [our documentation on accessing GPUs from Apptainer images](./apptainer.md#utilizing-gpu-resources-with-apptainer-images) for more information. In principle, the MPI installations can be also be accessed from correctly configured containers, but care is also needed to ensure compatibility between the libraries on the host and container.


## Building software images
Regardless of the runtime platform, images are built from a special configuration file. A `Dockerfile` is such a configuration for Docker, while Apptainer uses a "Definition File" (with a `.def` extension). These files specify the installation routines necessary to create the desired application, as well as any additional software packages to install and configure in this environment that may be required. You can think of these files as "recipes" for installing a given application you wish to containerize.

Building Docker or Apptainer images requires root/admin privileges and cannot be done directly by users of HPC systems. Docker is available on most platforms, and users with admin privileges on a local machine (such as your laptop) can build Docker images locally. The Docker image file can then be pushed to a registry and pulled on the HPC system using Apptainer as described [here](registries.md), or a tool such as [Docker2Singularity](https://github.com/singularityhub/docker2singularity) may be used to convert the image to the Apptainer format. Alternatively, users with admin privileges on a Linux system can run Apptainer locally to build images. Another option is to use Sylab's remote building [Container Service](https://cloud.sylabs.io/), which provides free accounts with a limited amount of build time for Apptainer-formatted images.

### Example Docker build workflow for HPC users

Because of the permission limitations described above, it is recommended that HPC users start with building a Docker image locally, e.g., on your laptop. If you are a researcher at NREL and plan to regularly containerize applications, you can request Docker to be installed at the admin-level on your work computer from the [IT Service Portal](https://nrel.servicenowservices.com). This section will describe a simple workflow for building a Docker image locally, exporting it as a `.tar` file, uploading it to Kestrel, and converting it to an Apptainer image for execution on HPC.

#### 1. Local Docker build

The following [Dockerfile](https://github.com/NREL/HPC/tree/master/docker-examples/simple-python3/Dockerfile) illustrates the build steps to create a small image. In this example, we simply install `python3` into an image based on the Ubuntu operating system (version 22.04):

```
# Docker example: save as `Dockerfile` in your working directory

FROM ubuntu:22.04

RUN apt-get update -y && apt-get install python3 -y
```

Images are normally built (or "bootstrapped") from a base image indicated by `FROM`. This base image is composed of one or more layers that will be pulled from the appropriate [container registry](registries.md) during buildtime. In this example, version 22.04 of the Ubuntu operating system is specified as the base image. Docker pulls from Ubuntu's [DockerHub container registry](https://hub.docker.com/_/ubuntu) by default. The ability to use a different base image provides a way to use packages which may work more easily on a specific operating system distribution. For example, the Linux distribution on Kestrel is Red Hat, so building the above image would allow the user to install packages from Ubuntu repositories.

The `RUN` portion of the above Dockerfile indicates the command to run *during the image's buildtime*. In this example, it installs the Python 3 package. Additional commands such as `COPY`, `ENV`, and others enable the customization of your image to suit your compute environment requirements. 

To build an image from the above Dockerfile (we will call it "simple_python3"), copy its contents to a file named `Dockerfile` in your current working directory and run the following:

```
docker build . -t simple_python3 --platform=linux/amd64
```

It is important to note that without the `--platform` option, `docker build` will create an image that matches your local machine's CPU chip architecture by default. If you have a machine running on `x86-64`/`amd64`, the container's architecture will be compatible NREL's HPC systems. If your computer does not use chips like these (such as if you have a Mac computer that runs on "Apple Silicon", which uses `arm64`), your image's architecture will *not* match what is found on NREL's HPC systems, causing performance degradation of its containers (at best) or fatal errors (at worst) during runtime on Kestrel, Swift, or Vermillion. Regardless of your local machine, as a best practice, you should explicitly specify your image's desired platform during buildtime with `--platform=linux/amd64` to ensure compatibility on NREL's HPC systems.

#### 2. Export Docker image to .tar

*Coming soon: a centralized software image registry/repository for NREL users, which will simplify the following steps. In the meantime, please follow steps 2 and 3 as written.* 

Once the Docker image is built, you can export it to a `.tar` archive with the following command:

```
docker image save simple_python3 -o simple_python3.tar
```

Depending on the specific application you are building, exported images can be relatively large (up to tens of GB). For this reason, you may wish to gzip/compress the `.tar` to a `.tar.gz`, which will save network bandwidth and ultimately reduce total transfer time:

```
tar czf simple_python3.tar.gz simple_python3.tar
```

#### 3. Upload exported image in `.tar.gz` format to HPC system

Now that the exported Docker image is compressed to `.tar.gz` format, you will need to transfer it to one of NREL's HPC systems. Considering the [scratch space](../../Systems/Kestrel/filesystems.md#scratchfs) of [Kestrel](../../Systems/Kestrel/getting_started_kestrel.md) as an example destination, we will use `rsync` as the transfer method. Be sure to replace `USERNAME` with your unique HPC username:

```
rsync -aP --no-g simple_python3.tar.gz USERNAME@kestrel.hpc.nrel.gov:/scratch/USERNAME/
```

For more information on alternatives to `rsync` (such as FileZilla or Globus), please refer to our [documentation regarding file transfers](../../Managing_Data/Transferring_Files/file-transfers.md).

#### 4. Convert .tar to Apptainer image

Once `rsync` finishes, you should find the following file (roughly 72MB in size) in your personal scratch folder on Kestrel (i.e., `/scratch/$USER`):

```
[USERNAME@kl1 USERNAME]$ ls -lh /scratch/$USER/simple_python3.tar.gz
-rw-r--r-- 1 USERNAME USERNAME 72M Mar 20 15:39 /scratch/USERNAME/simple_python3.tar.gz
```

The next step is to convert this "Docker archive" to an Apptainer-compatible image. Especially for larger images, this can be a memory-intensive process, so we will first [request a job from Slurm](../../Slurm/index.md), e.g.:

```
salloc -A <account> -p <partition> -t <time> ...
```

You can now convert the Docker image archive to an Apptainer `.sif` image on Kestrel with the following `build` command. Be sure to first unzip the `.tar.gz` archive, and prefix the resulting `.tar` with `docker-archive://`:

```
cd /scratch/$USER
module load apptainer/1.1.9
tar xzf simple_python3.tar.gz simple_python3.tar
apptainer build simple_python3.sif docker-archive://simple_python3.tar
```

Once this finishes, you can invoke the container with `apptainer exec simple_python3.sif <command>`. Anything that follows the name of the image will be executed *from the container*, even if the same command is found on the host system. To illustrate, if we examine the location of the `python3` binary within the `simple_python3.sif` image and the host system (Kestrel), we see they are both called from the location `/usr/bin/python3`:

```
# host's Python3
[USERNAME@COMPUTE_NODE USERNAME]$ which python3
/usr/bin/python3

# container's Python3
[USERNAME@COMPUTE_NODE USERNAME]$ apptainer exec simple_python3.sif which python3
/usr/bin/python3
```

However, the `apt-get install python3` command in the Dockerfile should have installed the most up-to-date Python3 library from Ubuntu's package manager, which is 3.10.12 at the time this is written. By contrast, the Python3 library installed on the host is older (version 3.6.8). In this way, we can confirm that the `python3` executed with `apptainer exec ...` is indeed originating from `simple_python3.sif`:

```
# host Python3
[USERNAME@COMPUTE_NODE USERNAME]$ python3 --version
Python 3.6.8

# container Python3
[USERNAME@COMPUTE_NODE USERNAME]$ apptainer exec simple_python3.sif python3 --version
Python 3.10.12
```

For more specific information on and best practices for using Apptainer on NREL's HPC systems, please refer to its [dedicated documentation page](./apptainer.md).

#### 5. A more involved Dockerfile example (CUDA 12.4)

For an example of an image you can build to provide everything needed for CUDA v.12.4, please refer to this [Dockerfile](https://github.com/NREL/HPC/tree/master/docker-examples/cuda-12.4/Dockerfile).

### Using Apptainer as build alternatives to Docker

Given Docker's popularity, support, and its widespread compatibility with other container runtimes, it is recommended to start your containerization journey with the steps outlined in the previous section. However, there could be rare cases in which you need to directly build an image with [Apptainer](https://apptainer.org/docs/user/main/definition_files.html). Instead of "Dockerfiles", these container runtimes use "Definition Files" for image building that have a similar, yet distinct format. Please refer to the respective link for more information. We also provide an [Apptainer image build example](./apptainer.md#create-ubuntu-based-image-with-mpi-support) in our documentation, which can be remotely built via the [Singularity Container Service](https://cloud.sylabs.io/) from Sylabs, the developer of Apptainer.
