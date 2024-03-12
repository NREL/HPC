---
layout: default
title: Containers Intro
parent: Containers
order: 1
---

# Introduction to software containerization

## What are software images/containers?
Software *images* provide a method of packaging your code so that it can be run anywhere you have a *container runtime environment*. This enables you to create an image on your local laptop and then run it on an HPC system or other computing resource. Software containerization provides an alternative way of isolating and packaging your code compared to solutions such as Conda virtual environments. 

**A note on terminology**: A software *container* is considered an instance of an *image*, meaning the former gets created during the runtime of the latter. In other words, a software *image* is what you build and distribute, whereas the *container* is what gets executed from a given image.

## Docker vs. Singularity/Apptainer
The most common container runtime environment (outside of HPC) is Docker. Due to the fact that it requires root-level permissions to build its associated images and run containers, Docker is not suited for HPC environments and is therefore not available on NREL's systems currently. Singularity, as well as its later release, Apptainer, are both alternative containerization tools that can be used in HPC environments because they do not require root. However, you can use Docker to build images locally and convert them to the Singularity/Apptainer format for use with HPC (described in more detail [here](index.md#building)).

Going forward in this documentation, the term "Apptainer" will be used to refer to both "Singularity" and "Apptainer" for brevity, unless otherwise specified. 

## Compatibility 
Apptainer is able to run most Docker images, but Docker is unable to run Apptainer images. A key consideration when deciding to containerize an application is which container engine to build with. A suggested best practice is to build images with Docker whenever possible, as this provides more flexibility. However, sometimes this is not feasible, and you may have to build with Apptainer or maintain separate images for each container engine.



## Advantages to software containerization
* **Portability**: Containers can be run on HPC, locally, and on cloud infrastructure used at NREL. 
* **Reproducibility**: Containers are one option to ensure reproducible research by packaging all necessary software to reproduce an analysis. Containers are also easily versioned using a hash.
* **Modularity**: Images are composed of cacheable "layers" of other images, facilitating the build process.
* **Workflow integration**: Workflow management systems such as Airflow, Nextflow, Luigi, and others provide built-in integration with container engines. 

## Accessing HPC hardware from software containers
Both Singularity and Docker provide the ability to use hardware based features on the HPC systems such as GPUs. A common usage for containers is packaging of GPU enabled tools such as TensorFlow. Singularity enables access to the GPU and driver on the host. In principle, the MPI installations can be also be accessed from correctly configured containers, but care is needed to ensure compatibility between the libraries on the host and container.

## Building
Software images are built from a special configuration file. A `Dockerfile` is such a configuration for Docker, while Apptainer uses a "Definition File" (with a `.def` extension). These files specify the installation routines necessary to create the desired application, as well as any additional software packages to install and configure in this environment that may be required.
```
# Docker example: save as `Dockerfile` in your working directory

FROM ubuntu:22.04

RUN apt-get update -y && apt-get install python3 -y
```

The above Dockerfile illustrates the build steps to create a simple image. Images are normally built (or "bootstrapped") from a base image indicated by `FROM`. This base image is composed of one or more layers that will be pulled from the appropriate [container registry](registries.md) during buildtime. In this example, version 22.04 of the Ubuntu operating system is specified as the base image. Docker pulls from ubuntu's [DockerHub container registry](https://hub.docker.com/_/ubuntu) by default. The ability to use a different base image provides a way to use packages which may work more easily on a specific operating system distribution. For example, the Linux distribution on Eagle is CentOS, so building the above image would allow the user to install packages from Ubuntu repositories.

The `RUN` portion of the above Dockerfile indicates the command to run *during the image's buildtime*. In this example, it installs the Python 3 package. Additional commands such as `COPY`, `ENV`, and others enable the customization of your image to suit your compute environment requirements. 

To build an image called "simple_python3" from the above Dockerfile and associate it with the "my_repo" DockerHub repository, copy its contents to a file named `Dockerfile` in your current working directory and run the following:

```
docker build . -t my_repo/simple_python3 --platform=linux/amd64
```

It is important to note that without the `--platform` option, `docker build` will create an image that matches your local machine's CPU chip architecture by default. If you have a machine running on `x86_64` or `amd` (such as most Intel chips), the container's architecture will be compatible NREL's HPC systems. If your computer does not use chips like these (such as if you have a Mac computer that runs on "Apple Silicon", which uses `arm`), your image's architecture will *not* match what is found on NREL's HPC systems, causing performance degradation of its containers (at best) or fatal errors (at worst) during runtime on Eagle, Kestrel, Swift, or Vermillion. Regardless of your local machine, as a best practice, you should explicitly specify your image's desired platform during buildtime with `--platform=linux/amd64` to ensure compatibility on NREL's HPC systems.

[Singularity](https://docs.sylabs.io/guides/latest/user-guide/definition_files.html) and [Apptainer](https://apptainer.org/docs/user/main/definition_files.html) definition files have a similar, yet distinct format. Please refer to the respective links for more information.

Building Docker or Singularity images requires root/admin privileges and cannot be done directly by users of HPC systems. Docker is available on most platforms, and users with admin privileges on a local machine (such as your laptop) can build Docker images locally. The Docker image file can then be pushed to a registry and pulled on the HPC system using Singularity as described [here](registries.md), or a tool such as [Docker2Singularity](https://github.com/singularityhub/docker2singularity) may be used to convert the image to a Singularity format. Alternatively, users with admin privileges on a Linux system can run Singularity locally to build images. Another option is to use [Singularity Container Service](https://cloud.sylabs.io/), which provides free accounts with a limited amount of container build time.