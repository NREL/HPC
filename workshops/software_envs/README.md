# NREL HPC Workshops: Software Environments on Eagle

Getting the software or analysis tools you need for your work can be a challenge. This workshop will discuss and demonstrate three common ways of getting your software environment set up on Eagle. Environment modules, Conda, and containers all have associated pros and cons which will be overviewed.

We will provide a background of how each technology works and common challenges. Effectively managing the software you use can greatly reduce the barriers to running your analysis, promote the portability of your work, and in some cases, speed it up!

## Modules

List all the available modules you can load on Eagle
```
ml avail 
```
List the currently loaded modules. This will currently be empty.
```
ml list
```

Now we will load a module, for example the GCC compiler.
```
ml gcc
```

If you hit Tab you will see the available versions such as: 
```
>ml gcc
gcc         gcc/10.1.0  gcc/5.5.0   gcc/6.5.0   gcc/7.4.0   gcc/8.4.0   gcc/9.3.0
```

Rerunning the list command will now show GCC has been loaded:
```
ml list
```
You should now see something like:
```
Currently Loaded Modules:
  1) gcc/10.1.0
```

To see what the modulefile for a given module, for example GCC again, contains you can use:
```
ml show gcc
```
The two most important components of the modulefile are `setenv` which sets environment variables and `prepend_path` which specifies paths which will be added to your `PATH`. 

Now we will take a look at some basics of what is happening when a module is loaded.

First we will unload all the loaded modules
```
module purge
```

Next we will run a number of commands to show how changes are happening. First we will `echo` both the standard `$PATH` variable which shows where binaries are searched for and an environment variable `CC`. First, the `CC` variable will likely be empty depending on your environment. The `PATH` will print a number of existing paths for your environment. Once `ml` has been run, you should now see a string when you echo `CC` and additional paths at the start of your `PATH`.
```
echo $CC
echo $PATH
ml gcc
echo $PATH
echo $CC
```

Another useful feature of modules is the ability to load multiple modules and save this as a set which can be easily loaded. The following example loads GCC and OpenMPI and saves this collection as `myproject`. 

```
ml gcc
ml openmpi/4.1.0
ml save myproject
ml describe myproject
```

Once you have a collection saved you can load it again:
```
ml restore myproject
```
**Additional resources**

* [Lmod docs](https://lmod.readthedocs.io/en/latest/010_user.html)
* [NREL modules docs](https://www.nrel.gov/hpc/eagle-environment-modules.html)

## Conda

```
ml conda
```

Now we can create a Conda environment with a specified version of Python. 
```
conda create -n workshop python=3.8
```
OR
```
mamba create -n workshop python=3.8
```

Conda is a package manager too so we can use `install` to add new packages to our environment. 
```
conda install numpy
```
OR
```
mamba install numpy
```

Similar to how Modules modify your `PATH` environment variable Conda also appends directories to this variable. Running the following series of commands will show that Conda is adding a directory for the `workshop` environment to the `PATH`.
```
echo $PATH
conda activate workshop
echo $PATH
```

Mix and match with Modules is straightforward. The following commands will show that the Python version being used is the one installed by Conda, and then gcc is loaded. Both modules and Conda can be used together as long as the packages are not conflicting. 
```
which python
which gcc
ml gcc
which gcc
which python
```

**Additional Resources**
* [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
* [Conda docs](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)


## Containers
First we will start in Docker on a local device. 

We will compare what OS we see and then run a container and check again. 

```
echo $OSTYPE
docker run -it ubuntu
echo $OSTYPE
cat /etc/os-release
```

Below is a very minimal Docker recipe which is found in the [Dockerfile](./Dockerfile). The `FROM` line specifies the base image which you will build on. We then use the `RUN` command to run a pip install.  
```
FROM python:3
RUN pip install numpy
```
Now we will use the Singularity module to explore containers on Eagle.
```
ml singularity-container
```
We will pull a TensorFlow container with GPU support as our demo. Singularity is able to pull either Singularity or Docker images from repositories. We will pull the TensorFlow container from Dockerhub.

First we will search [Dockerhub](https://hub.docker.com/) for an appropriate container. 

**Note: this will pull a large file**

```
singularity pull docker://tensorflow/tensorflow:latest-gpu
```

We can now run the container:
```
singularity run tensorflow_latest-gpu.sif
```

As this is a GPU container we can also use `--nv` to enable the GPU inside the container.
```
singularity run --nv tensorflow_latest-gpu.sif
```

Now we are able to see the GPU in our container.

Another challenge with containers is managing how data is mounted into the container. Singularity by default will mount your home directory, other directories must be specified using `--bind`. For example, we can bind in our `scratch` directory which will be available at the path `/data` in the container. 

```
singularity run --nv --bind /scratch/$USER:/data tensorflow_latest-gpu.sif
```



**Additional Resources**

* [Singularity docs](https://sylabs.io/guides/3.1/user-guide/)
* [Docker docs](https://docs.docker.com/get-started/)
* [NREL Harbor Container Repository](https://harbor.nrel.gov)

