## How to use Apptainer (Singularity) on Kestrel

Singularity has been deprecated in favor of a new container runtime environment called Apptainer, which is its direct decendent. Apptainer will run Singularity containers and it supports Singularity commands by default. On Kestrel, `singularity` is an alias for `apptainer` and the two commands can be used interchangeably in most instances. However, since Singularity is deprecated, it is advised to use Apptainer.

More information about Apptainer can be found at [https://apptainer.org](https://apptainer.org). 

On Kestrel, Apptainer is installed on compute nodes and is accessed via a module named `apptainer` (you can check the current default module via `ml -d av apptainer`). The directory `/nopt/nrel/apps/software/apptainer/1.1.9/examples` holds a number of images (`*.sif`) and an example script (`script`) that shows how to run containers hosting MPI programs across multiple nodes. The `script` can also be accessed from [our GitHub repository](https://github.com/NREL/HPC/blob/master/kestrel/apptainer/script).

Before we get to the more complicated example from `script`, we'll first look at downloading (or *pulling*) and working with a simple image.

Input commands are preceded by a `$`.

!!! note
    If you wish to containerize your own application, it may be worth starting with [building a local Docker image and transferring it to Kestrel](./index.md#example-docker-build-workflow-for-hpc-users) before attempting to directly create your own Apptainer image, since you do not have root access on Kestrel.

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
[USERNAME@x1000c0s0b0n0 ~]$ ml -d av apptainer
------------------------------------ /nopt/nrel/apps/modules/default/application ------------------------------------
   apptainer/1.1.9
[USERNAME@x1000c0s0b0n0 ~]$ module load apptainer/1.1.9
```
Note: at the time of writing, `apptainer/1.1.9` is the default Apptainer module on Kestrel as determined by running `ml -d av apptainer`.

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

??? example "Sample job script: Running MPI-enabled Apptainer containers"
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

GPU-accelerated software often have complex software and hardware requirements to function properly, making containerization a particularly attractive option for deployment and use. These requirements manifest themselves as you are building your image (*buildtime*) and when you run a container (*runtime*). This section describes key components of software images that are successfully GPU-enabled with a Tensorflow container example. For more detailed documentation on the subject, visit Apptainer's dedicated [GPU Support page](https://apptainer.org/docs/user/1.0/gpu.html).


### Tensorflow Container Example

#### 1. Pull a compatible version of GPU-enabled Tensorflow from DockerHub

There are several versions (*tags*) of [Tensorflow images](https://hub.docker.com/r/tensorflow/tensorflow/tags) available from the DockerHub [container registry](../Documentation/Development/Containers/registries.md), each with different versions of GPU drivers and CUDA. You can obtain this information from the host by running the command `nvidia-smi` after allocating a GPU within a Slurm job on your desired system. Alternatively, you could simply consult the table below. If your running container is installed with a different GPU driver/CUDA version than what is listed below for your target system, you will either run into a fatal error, or the software will bypass the GPU and run on the CPU, slowing computation.

| System   | Partition name | GPU type<br>(cards per node) | `nvidia-smi`<br>GPU driver version | CUDA Version |
|:--------:|:--------------:|:----------------------------:|:----------------------------------:|:------------:|
| Kestrel  | gpu-h100       | H100 (4)                     | 550.54.15                          | 12.4         |
| Swift    | gpu            | A100 (4)                     | 550.54.15                          | 12.4         |
| Vermilion| gpu            | A100 (1)                     | 460.106.00                         | 11.2         |

Kestrel's H100 GPUs run with CUDA 12.4 with a GPU driver version of 550.54.15. Most GPU-enabled applications are compatible with a given major version release of CUDA; for example, if an application requires CUDA/12.4, it will more than likely work with other versions of CUDA >= 12.0. So for this example on Kestrel, we are looking for a Tensorflow image tag that includes as close to CUDA/12.4 as we can. On DockerHub, we see from consulting the [layers of `tensorflow:2.15.0-gpu`](https://hub.docker.com/layers/tensorflow/tensorflow/2.15.0-gpu/images/sha256-66b44c162783bb92ab6f44c1b38bcdfef70af20937089deb7bc20a4f3d7e5491?context=explore) that this image fits our requirements (note line 14: `ENV CUDA_VERSION=12.3.0`). *At the time of writing, a Tensorflow image with CUDA/12.4 is not yet available from this DockerHub repository.*


First, allocate a Kestrel GPU compute node:

```
salloc -A <YOUR-ACCOUNT> -t 1:00:00 --gpus=1 -N 1 -n 1 --mem-per-cpu=8G
```

!!! note
    We are only requesting 1 GPU card (`--gpus=1`) of the 4 available per node, and subsequently 1 task (`-n 1`). Though we are automatically given access to all of the *GPU memory* on the node, we request 8G of *CPU memory* from `salloc`. This is because our Tensorflow example will require a decent amount of CPU memory as it copies data to and from the GPU device. If such CPU memory is a bottleneck in a real-world example, you may want to consider replacing `-n 1 --mem-per-cpu=8G` with `--exclusive` to request all of the node's CPU resources, even if you are only using a single GPU card.


Once we are allocated a node, we will load the Apptainer module, and then pull `tensorflow:2.15.0-gpu` from DockerHub to a personal scratch location on Kestrel.

```
module load apptainer
apptainer pull /scratch/$USER/tensorflow-2.15.0.sif docker://tensorflow/tensorflow:2.15.0-gpu
```

Once the image finishes pulling, we can see a new `.sif` file in `/scratch/$USER`:

```
ls -lh /scratch/$USER/tensorflow-2.15.0.sif
-rwxrwxr-x 1 USERNAME USERNAME 3.4G Apr  3 11:49 /scratch/USERNAME/tensorflow-2.15.0.sif
```

!!! note
    We recommend saving `.sif` files to `/scratch` or `/projects` whenever feasible, as these images tend to be large, sometimes approaching tens of GB.

#### 2. Verify GPU device is found

##### Recognizing GPU device from Slurm
As a reminder, we only requested 1 GPU card in our `salloc` command above. On the Slurm side of things, we can verify this device is accessible to our computing environment by examining the contents of the `SLURM_GPUS_ON_NODE` (the number of allocated GPU cards) and `SLURM_JOB_GPUS` (the device's ID). By grepping for `GPU` from our list of environmental variables, we can see that Slurm indeed recognizes a single GPU device with ID `0`:

```
env | grep GPU
SLURM_GPUS_ON_NODE=1
SLURM_JOB_GPUS=0
```

##### Recognizing GPU device from the container
It is important to note that just because Slurm has allocated this device, it doesn't necessarily mean that the Tensorflow container can recognize it. Let's now verify that a GPU is accessible on the containerized Python side of things. We will invoke `/scratch/$USER/tensorflow-2.15.0.sif` to see whether Tensorflow itself can use the GPU allocated by Slurm:

```
apptainer exec /scratch/$USER/tensorflow-2.15.0.sif python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Oh no! You should see this error as the output from the above command:

```
libcuda reported version is: NOT_FOUND: was unable to find libcuda.so DSO loaded into this program
```

What happened here - didn't we pull a Tensorflow image that contains CUDA/12.3? We did, but whenever you run GPU-enabled Apptainers, **it is critical to supply the `--nv` flag after `exec`, otherwise the GPU device(s) will not be found**. You can read more about what `--nv` does [here](https://apptainer.org/docs/user/1.0/gpu.html#requirements).

Let's try finding this device from Python again, this time after supplying `--nv` to the container runtime:

```
apptainer exec --nv /scratch/$USER/tensorflow-2.15.0.sif python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

That's better! We can now see that GPU device `0` as allocated by Slurm is accessible to Tensorflow:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

#### 3. Run example Tensorflow training script

Now we are ready to run the Tensorflow container. We will run the script below, which is based on [Tensorflow's advanced quickstart example](https://www.tensorflow.org/tutorials/quickstart/advanced). This script tests a model that is trained on the mnist example dataset.

??? example "Python script: Simple GPU Tensorflow train and test"
    ```
    import time
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Flatten, Conv2D
    from tensorflow.keras import Model

    # source: https://www.tensorflow.org/tutorials/quickstart/advanced

    ### load mnist dataset
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    ### Use tf.data to batch and shuffle the dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    # Create an instance of the model
    model = MyModel()

    ### optimizer/loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    ### Select metrics to measure the loss and the accuracy of the model. These metrics accumulate the values over epochs and then print the overall result.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    ### train the model
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # test the model
    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    t0 = time.time()
    EPOCHS = 10
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
    t1 = time.time()
    print(f'A total of {EPOCHS} epochs took {t1-t0} seconds')
    ```

Save this script as `tensor_test.py` into your current working directory and run the following command:

```
apptainer exec --nv /scratch/$USER/tensorflow-2.15.0.sif python tensor_test.py
```

Assuming you made the same `salloc` request above, it should take ~26 seconds to run through 10 training/testing epochs on a single GPU. This particular container is sophisticated enough to automatically switch between CPU and GPU computation depending on the availability of a GPU device. If you'd like to compare the time it takes for this script to run purely on a CPU, simply omit the `--nv` flag from your call to `apptainer` above and run the command on the same node. You should observe that the runtime jumps to ~252 seconds, meaning that the GPU computation is almost 10 times faster than the CPU!


## Best practices and recommendations

This section describes general recommendations and best practices for Apptainer users across NREL's HPC systems.

### Change Apptainer cache location to `/scratch/$USER`

By default, Apptainer will cache image layers to your `$HOME` folder when you pull or build `.sif` images, which is not ideal as users have a limited storage quota in `/home`. As you continue to use Apptainer, this cache folder can become quite large and can easily fill your `$HOME`. Fortunately, the location of this cache folder can be controlled through the `APPTAINER_CACHEDIR` environmental variable. To avoid overfilling your `$HOME` with unnecessary cached data, it is recommended to add an `APPTAINER_CACHEDIR` location to your `~/.bashrc` file. You can accomplish this with the following command, which will direct these layers to save to a given system's scratch space:

`echo "export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer" >> ~/.bashrc`

Note that you will either need to log out and back into the system, or run `source ~/.bashrc` for the above change to take effect.

### Save `.def` files to home folder and images to /scratch or /projects

An Apptainer definition file (`.def`) is a relatively small text file that contains much (if not all) of the build context for a given image. Since your `$HOME` folders on NREL's HPC systems are regularly backed up, it is strongly recommended to save this file to your home directory in case it accidentally gets deleted or otherwise lost. Since `.sif` images themselves are 1. typically large and 2. can be rebuilt from the `.def` files, we recommend saving them to a folder outside of your `$HOME`, for similar reasons described in the previous section. If you intend to work with an image briefly or intermittantly, it may make sense to save the `.sif` to your `/scratch` folder, from which files can be purged if they haven't been accessed for 28 days. If you plan to use an image frequently over time or share it with other users in your allocation, saving it in a `/projects` location you have access to may be better.


### Bind Mounting Directories

By default, most containers only mount your `$HOME` folder, current working directory, and a handful of other common folders. If a host directory isn't in this list and isn't explicitly provided during runtime, you may get a "File not found" error. For example, if you are running a container from `/scratch/$USER` and want to write a result file to a `/projects` location, you will need to provide the mount path with the `-B </path/on/host>:</path/in/container>` option:

```
apptainer -B /projects:/projects --nv exec IMAGE.sif COMMAND > /projects/my-project/result.txt
```

### Provide the `--nv` flag to Apptainer Runtime (GPU)

Once you allocate at least one GPU card in your job, you then need to make Apptainer recognize the GPU resources you wish to use. To accomplish this, you can supply the `--nv` flag to an `apptainer shell ...` or `apptainer exec ...` command. Using a generic `gpu_accelerated_tensorflow.sif` image as an example:

```
apptainer exec --nv gpu_accelerated_tensorflow.sif python tensorflow.py
```

### Providing CUDA to Host Environment (GPU)
In the [Tensorflow example above](./apptainer.md#tensorflow-container-example), the container included all of the necessary software to run on a GPU, including CUDA. However, depending on the specific software container you are trying to run, its image may or may not include a working version of CUDA. If you encounter CUDA- or GPU-driver errors, try loading version 12.4 of the CUDA module before running the container:

```
module load cuda/12.4
```



