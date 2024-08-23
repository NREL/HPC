FROM ubuntu:22.04

## This is the base image for CUDA 12.4 that subsequent NVIDIA images run at NREL rely upon.
## Contact: Matt.Selensky@nrel.gov

#### Specifications
#### OS: Ubuntu v. 22.04
#### CUDA: 12.4
#### CUDNN: 9

# syntax for NVIDIA package repos
ARG distro=ubuntu2204
ARG arch=x86_64

ENV DEBIAN_FRONTEND=noninteractive

### Install Packages and Libraries ###

## Compilers and Fetchers
RUN apt-get update -y && apt-get install -y wget git ca-certificates libssl-dev
RUN apt-get update -y && apt-get install -y build-essential cmake

## Cuda/Cudatoolkit
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.pin && \
  mkdir -p /etc/apt/preferences.d/ && \
  mv cuda-$distro.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
  wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-$distro-12-4-local_12.4.0-550.54.14-1_amd64.deb
RUN dpkg -i cuda-repo-$distro-12-4-local_12.4.0-550.54.14-1_amd64.deb && \
  cp /var/cuda-repo-$distro-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
  apt-get update -y && apt-get -y install cuda-toolkit-12-4
# cudnn
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update -y && apt-get install -y cudnn9-cuda-12

# cleanup
RUN rm -rf /cuda-repo-$distro-12-4-local_12.4.0-550.54.14-1_amd64.deb /cuda-keyring_1.1-1_all.deb 

# Python, pip, and venv (add specific packages for individual tests)
RUN apt-get install python3 python3-venv -y
RUN ln -s `which python3` /usr/bin/python

# install pip this way due to https://github.com/pypa/setuptools/issues/3269
RUN wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py && python get-pip.py

# final things
ENV CUDA_HOME=/usr/local/cuda
ENV CUDNN_INCLUDE_DIR=/usr/include
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=/usr/lib:/usr/lib64:/usr/local/lib:/usr/local/cuda/lib64
