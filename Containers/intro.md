---
layout: default
title: Containers Intro
parent: Containers
---

# Introduction to containers

## What are containers?
Containers provide a method of packaging your code so that it can be run anywhere you have a container runtime. This enables you to create a container on your local laptop and then run it on Eagle or other computing resources. Containers provide an alternative way of isolating and packaging your code from solutions such as Conda environments. 

## Docker vs. Singularity
The most common container runtime outside of HPC is Docker. Docker is not suited for the HPC environment on Eagle as is therefore not available on the system currently. Singularity is an alternative container tool which is provided. 

## Compatibility 
Singularity is able to run most Docker images, but Docker is unable to run Singularity images. A key consideration when deciding to containerize an application is which container engine to build with. The suggested best idea

## Building
Containers are built from a container specification file, Dockerfiles for Docker or Singularity Definition File in Singularity. These files specify the steps necessary to create the desired package and the additional software packages to install and configure in this environment. 
```

```




## Container advantages
* **Portability**: containers can be run on HPC, locally, and on cloud infrastructure used at NREL. 
* **Reproducibility**: Containers are one option to ensure reproducible research by packaging all necessary software to reproduce an analysis. Containers are also easily versioned using a hash.
* **Workflow integration**: Workflow management systems such as Airflow, Nextflow, Luigi, and others provide built in integration with container engines. 

## HPC hardware
Both Singularity and Docker provide the ability to use hardware based features of Eagle. 