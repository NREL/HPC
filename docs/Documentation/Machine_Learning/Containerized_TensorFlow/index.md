---
layout: default
title: Containerized TensorFlow
parent: Containers
---
## TensorFlow with GPU support - Apptainer
This Apptainer image supplies TensorFlow 2.15.0 optimized for use with GPU nodes running CUDA > 12.3 (which works with Kestrel's H100s). It also includes opencv, numpy, pandas, seaborn, scikit-learn, and a number of other Python libraries. More information about Tensorflow's containerized images can be found on [DockerHub](https://hub.docker.com/r/tensorflow/tensorflow/tags).

For more information on Apptainer in general, on please see: [Containers](../../Development/Containers/index.md).

### Quickstart

After allocating a job, note that you will have to bind mount `/nopt` (where the image lives) as well as the parent directory of where you are working from (e.g., `/scratch` or `/projects`)

```bash
# Get allocation
salloc --gres=gpu:2 -N 1 --mem=80G -n 32 -A <allocation handle> -t 01:00:00 -p debug
# Run Apptainer in srun environment
module load apptainer
# Note that you will have to bind mount /nopt (where the image lives) as well as the parent directory of where you are working from (e.g., /scratch or /projects)
cd /projects/<MY_HPC_PROJECT>
srun --gpus=2 --pty apptainer shell -B /nopt:/nopt -B /projects:/projects --nv /nopt/nrel/apps/gpu_stack/ai_substack/tensorflow-2.17.0-gpu-jupyter.sif
```

### Building a custom image based on TensorFlow
In order to build a custom Apptainer image based on this one, Docker must be installed on your local computer. Please refer to [our example Docker build workflow for HPC users](../../Development/Containers/index.md#example-docker-build-workflow-for-hpc-users) for more information on how to get started.

This workflow is useful if you need to modify the prebuilt Tensorflow image for your own purposes (such as if you need extra Python libraries to be available). You can copy a `requirements.txt` into the container during buildtime and upload the resulting image to Kestrel, where it can be converted to Apptainer format for runtime.

1. Update Dockerfile shown below to represent the changes desired and save to working directory.  
```
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter
ENV DEBIAN_FRONTEND="noninteractive" 
RUN apt-get -y update
RUN apt-get -y install python3-opencv
RUN mkdir /custom_env
COPY requirements.txt /custom_env
RUN pip install -r /custom_env/requirements.txt
```
2. Update `requirements.txt` shown below for changing the python library list and save to working directory.
```
seaborn
pandas
numpy
scikit-learn
git+https://github.com/tensorflow/docs
```
3. Build new Docker image for x86_64.
```bash
docker build -t tensorflow-custom-tag-name . --platform=linux/amd64
```
4. Follow [the instructions here](../../Development/Containers/index.md#example-docker-build-workflow-for-hpc-users) for exporting the Docker image to a `.tar` archive, uploading it to Kestrel, and using Apptainer to convert it to Apptainer format to run on HPC.