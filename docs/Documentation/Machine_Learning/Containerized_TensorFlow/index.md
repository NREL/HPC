---
layout: default
title: Containerized TensorFlow
parent: Containers
---
## TensorFlow with GPU support singularity container
This Singularity container supplies TensorFlow 2.3.0 optimized for use with GPU nodes.  It also has opencv, numpy, pandas, seaborn, scikit-learn python libraries.

For more information on Singularity on please see: [Containers](../../Development/Containers/index.md)
### Quickstart
```bash
# Get allocation
salloc --gres=gpu:2 -N 1 -A hpcapps -t 0:10:00 -p debug
# Run singularity in srun environment
module load singularity-container
unset LD_PRELOAD
srun --gpus=2 --pty singularity shell --nv /nopt/nrel/apps/singularity/images/tensorflow_gpu_extras_2.3.0.sif
```

### Building a custom image based on TensorFlow
In order to build a custom Singularity image based on this one, docker must be installed on your local computer.  [Docker documentation](https://docs.docker.com/engine/install/) shows how to install docker.

1. Update Dockerfile shown below to represent the changes desired and save to working directory.  
```
FROM tensorflow/tensorflow:2.3.0-gpu-jupyter
RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install python3-opencv
RUN mkdir /custom_env
COPY requirements.txt /custom_env
RUN pip install -r /custom_env/requirements.txt
```
2. Update requirements.txt shown below for changing the python library list and save to working directory.
```
seaborn
pandas
numpy
scikit-learn
git+https://github.com/tensorflow/docs
```
3. Build new docker image
```bash
docker build -t tensorflow-custom-tag-name .
```
4. Create Singularity image file.  Note the ./images directory must be created before running this command.
```bash
docker run -v /var/run/docker.sock:/var/run/docker.sock \
-v $(PWD)/images:/output \
--privileged -t --rm \
quay.io/singularity/docker2singularity --name tensorflow_custom.sif \
tensorflow-custom-tag-name
```
5. Transfer image file to Eagle.  For this example I created a directory named /scratch/$(USER)/tensorflow on eagle
```bash
rsync -v images/tensorflow_custom.sif eagle.hpc.nrel.gov:/scratch/$(USER)/tensorflow/
```
