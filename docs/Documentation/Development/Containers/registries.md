# Container registries at NREL

## Introduction
Container registries enable users to store container images. An overview of the steps to use each fo the main container registries available to NREL users is provided below. Registries can enable reproducibility by storing tagged versions of containers, and also facilitate transferring images easily between different computational resources. 

## Create Docker images
Docker is not supported on NREL's HPC systems, including Kestrel. Instead, [Apptainer](apptainer.md) is the container engine provided as a module. Apptainer is able to pull Docker images and convert them to Apptainer-formatted images. We generally recommend [building Docker images](index.md#example-docker-build-workflow-for-hpc-users) to ensure portability between compute resources and using Apptainer to convert the image when running on an HPC system. 

## Accessibility

| Registry  | Kestrel Access | AWS Access | Docker Support | Apptainer Support   |
| --------- | -------------- | ---------- | -------------- | ------------------- |
| Harbor    | No**           | No         | Yes            | Yes                 |
| AWS ECR   | Yes            | Yes        | Yes            | No*                 |
| DockerHub | Yes            | Yes        | Yes            | No*                 |
*for DockerHub and AWS ECR it may be possible to push images using ORAS, but this was not found to be a streamlined process in testing. 
**Harbor was originally set up for Kestrel's predecessor, Eagle. A replacement is being identified.

## AWS ECR
AWS ECR can be utilized by projects with a cloud allocation to host containers. ECR primarily can be used with Docker containers, although Apptainer should also be possible. 

## Harbor
[NREL's Harbor](https://harbor.nrel.gov) is a registry hosted by ITS that supports both Docker and Apptainer containers. Harbor was originally set up for Kestrel's predecessor, Eagle, which also used Apptainer's predecessor, Singularity. ****NREL ITS is currently evaluating a replacement to internally hosted Harbor (likely moving to Enterprise [DockerHub](#dockerhub))** The following information is archived until such a replacement is identified for Kestrel.

### Docker
#### Login
On your local machine to push a container to the registry. 
```
docker login harbor.nrel.gov
```

#### Prepare image for push

```
docker tag SOURCE_IMAGE[:TAG] harbor.nrel.gov/REPO/IMAGE[:TAG]
```

```
docker push harbor.nrel.gov/REPO/IMAGE[:TAG]
```

#### Pull Docker image on Eagle
Pull and convert container to Singularity on Eagle.

**Note:** `--nohttps` is not optimal but need to add certs for NREL otherwise there is a cert error. 
```
apptainer pull --nohttps --docker-login docker://harbor.nrel.gov/REPO/IMAGE[:TAG]
```

**The container should now be downloaded and usable as usual**

### Singularity
#### Login information
Under your User Profile in Harbor obtain and export the following information
```
export SINGULARITY_DOCKER_USERNAME=<harbor username>
export SINGULARITY_DOCKER_PASSWORD=<harbor CLI secret>
```

#### Push a Singularity image
```
singularity push <image>.sif oras://harbor.nrel.gov/<PROJECT>/<IMAGE>:<TAG>
```

#### Pull a Singularity image
```
singularity pull oras://harbor.nrel.gov/<PROJECT>/<IMAGE>:<TAG>
```

## DockerHub

**An enterprise version of DockerHub is being evaluated and is currently unavailable.** However, NREL HPC users are free to pull Docker images with Apptainer directly from the public version of DockerHub. For example, this pulls the official Ubuntu v.22.04 image from DockerHub and converts it to the Apptainer-formatted `ubuntu-22.04.sif` image:

```
apptainer pull ubuntu-22.04.sif docker://ubuntu:22.04
```

!!! Note 
    DockerHub maintains a series of ["official" images](https://hub.docker.com/search?image_filter=official) that follow the syntax `apptainer pull <name of SIF> docker://<image name>:<image version>` when pulling with Apptainer. For all other images that are not listed in the link, you should instead use the syntax `apptainer pull <name of SIF> docker://<image repo name>/<image name>:<image version>`.

### DockerHub Enterprise Credentials
To get the needed credentials for NREL Dockerhub, select your username in the top right -> Account -> Security -> Create a new access token.

The dialog box will describe how to use the security token with `docker login` to enable pulling and pushing containers. 
