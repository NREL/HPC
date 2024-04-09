# Container registries at NREL

## Introduction
Container registries enable users to store container images. An overview of the steps to use each fo the main container registries available to NREL users is provided below. Registries can enable reproducibility by storing tagged versions of containers, and also facilitate transferring images easily between different computational resources. 

## Create Docker images
Docker is not supported on NREL's HPC systems including Eagle. Instead Singularity is the container engine provided as a module. Singularity is able to pull Docker images and convert them to Singularity images. Although not always possible, we suggest creating Docker images when possible to ensure portability between compute resources and using Singularity to convert the image if it is to be run on an HPC system. 

## Accessibility

| Registry | Eagle Access | AWS Access | Docker Support | Singularity Support |
| -------- | ------------ | ---------- | -------------- | ------------------- |
| Harbor   | Yes          | No         | Yes            | Yes                 |
| AWS ECR  | Yes          | Yes        | Yes            | No*                 |
| DockerHub | Yes         | Yes        | Yes            | No*                 |
*for DockerHub and AWS ECR it may be possible to push images using ORAS, but this was not found to be a streamlined process in testing. 

## AWS ECR
AWS ECR can be utilized by projects with a cloud allocation to host containers. ECR primarily can be used with Docker containers, although Singularity should also be possible. 

## Harbor
[NREL's Harbor](https://harbor.nrel.gov) is a registry hosted by ITS that supports both Docker and Singularity containers. 

****NREL ITS is currently evaluating a replacement to internally hosted Harbor (likely moving to Enterprise [DockerHub](#dockerhub))**

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
singularity pull --nohttps --docker-login docker://harbor.nrel.gov/REPO/IMAGE[:TAG]
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

## Dockerhub

**Currently under testing, and not generally available**

## Credentials
To get the needed credentials for NREL Dockerhub, select your username in the top right -> Account -> Security -> Create a new access token.

The dialog box will describe how to use the security token with `docker login` to enable pulling and pushing containers. 
