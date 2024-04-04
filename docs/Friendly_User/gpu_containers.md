[Software containerization](../Documentation/Development/Containers/index.md) is a robust method of deploying reproducible code that may have complex software and/or hardware requirements. For this reason, many applications (including those that are GPU-accelerated) can greatly benefit from containerization. This page describes key considerations for effectively using a GPU resource from a software container during its runtime on Kestrel.

## Terminology

In the parlance of software containers, the *host* is the computer/node on which you execute a *container*, the latter of which is an instance of a software *image*.

## Allocate a GPU Job

As with any GPU-accelerated software, you first need to ensure you have the right hardware accessible. On Kestrel, this means you need to start with [allocating a job that provides at least one GPU card](running_gpu.md) to the compute node host's computing environment before continuing.

Once you have such a job allocated, we will illustrate key considerations for a typical GPU container use case by describing the steps required to run a Tensorflow software image that is compatible with the GPU drivers available on Kestrel. The example we will use will run on a single GPU card, so your (interactive) Slurm job allocation command may look something like:

```
salloc -A <YOUR-ACCOUNT> -p gpu-h100 -t 1:00:00 --gres=gpu:1 -N 1 -n 1 --mem-per-cpu=8G
```

!!! note
    We are only requesting 1 GPU card (`--gres=gpu:1`) of the 4 available per node, and subsequently 1 task (`-n 1`). Though we are automatically given access to all of the *GPU memory* on the node, we request 8G of *CPU memory* from `salloc`. This is because our Tensorflow example will require a decent amount of CPU memory as it copies data to and from the GPU device. If such CPU memory is a bottleneck in a real-world example, you may want to consider replacing `-n 1 --mem-per-cpu=8G` with `--exclusive` to request all of the node's CPU resources, even if you are only using a single GPU card.

## Tensorflow Container Example

### 1. Pull a compatible version of GPU-enabled Tensorflow from DockerHub

There are several versions (*tags*) of [Tensorflow images](https://hub.docker.com/r/tensorflow/tensorflow/tags) available from the DockerHub [container registry](../Documentation/Development/Containers/registries.md), each with different versions of GPU drivers and CUDA. To successfully use a GPU-enabled container, we need to ensure that these drivers and CUDA are compatible with what is available on our host node. Kestrel's H100 GPUs run with CUDA 12.3 with a GPU driver version of 545.23.08 (this information can be obtained by running `nvidia-smi` on the GPU node):

| System   | Partition name | GPU type<br>(cards per node) | `nvidia-smi`<br>GPU driver version | CUDA Version |
|:--------:|:--------------:|:----------------------------:|:----------------------------------:|:------------:|
| Kestrel  | gpu-h100       | H100 (4)                     | 545.23.08                          | 12.3         |

As such, we are ideally looking for a Tensorflow image tag that includes CUDA/12.3. On DockerHub, we see from consulting the [layers of `tensorflow:2.15.0-gpu`](https://hub.docker.com/layers/tensorflow/tensorflow/2.15.0-gpu/images/sha256-66b44c162783bb92ab6f44c1b38bcdfef70af20937089deb7bc20a4f3d7e5491?context=explore) that this image fits our requirements (note line 14: `ENV CUDA_VERSION=12.3.0`).

On a GPU compute node, we will first load the [Apptainer](../Documentation/Development/Containers/apptainer.md) module, and then pull `tensorflow:2.15.0-gpu` from DockerHub to a personal `${LOCAL_SCRATCH}` location on Kestrel.

```
module load apptainer
apptainer pull ${LOCAL_SCRATCH}/tensorflow-2.15.0.sif docker://tensorflow/tensorflow:2.15.0-gpu
```

Once the image finishes pulling, we can see a new `.sif` file in `${LOCAL_SCRATCH}`:

```
ls -lh ${LOCAL_SCRATCH}/tensorflow-2.15.0.sif
-rwxrwxr-x 1 USERNAME USERNAME 3.4G Apr  3 11:49 /scratch/USERNAME/tensorflow-2.15.0.sif
```

!!! note
    We recommend saving `.sif` files to `${LOCAL_SCRATCH}` or `/projects` whenever feasible, as these images tend to be large, sometimes approaching tens of GB.

### 2. Verify GPU device is found

#### Recognizing GPU device from Slurm
As a reminder, we only requested 1 GPU card in our `salloc` command above. On the Slurm side of things, we can verify this device is accessible to our computing environment by examining the contents of the `SLURM_GPUS_ON_NODE` (the number of allocated GPU cards) and `SLURM_JOB_GPUS` (the device's ID). By grepping for `GPU` from our list of environmental variables, we can see that Slurm indeed recognizes a single GPU device with ID `0`:

```
env | grep GPU
SLURM_GPUS_ON_NODE=1
SLURM_JOB_GPUS=0
```

#### Recognizing GPU device from the container
It is important to note that just because Slurm has allocated this device, it doesn't necessarily mean that the Tensorflow container can recognize it. Let's now verify that a GPU is accessible on the containerized Python side of things. We will invoke `${LOCAL_SCRATCH}/tensorflow-2.15.0.sif` to see whether Tensorflow itself can use the GPU allocated by Slurm:

```
apptainer exec ${LOCAL_SCRATCH}/tensorflow-2.15.0.sif python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Oh no! You should see this error as the output from the above command:

```
libcuda reported version is: NOT_FOUND: was unable to find libcuda.so DSO loaded into this program
```

What happened here - didn't we pull a Tensorflow image that contains CUDA/12.3? We did, but whenever you run GPU-enabled Apptainers, **it is critical to supply the `--nv` flag after `exec`, otherwise the GPU device(s) will not be found**. You can read more about what `--nv` does [here](https://apptainer.org/docs/user/1.0/gpu.html#requirements).

Let's try finding this device from Python again, this time after supplying `--nv` to the container runtime:

```
apptainer exec --nv ${LOCAL_SCRATCH}/tensorflow-2.15.0.sif python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

That's better! We can now see that GPU device `0` as allocated by Slurm is accessible to Tensorflow:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 3. Run example Tensorflow training script

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
apptainer exec --nv ${LOCAL_SCRATCH}/tensorflow-2.15.0.sif python tensor_test.py
```

Assuming you made the same `salloc` request [above](#allocate-a-gpu-job), it should take ~26 seconds to run through 10 training/testing epochs on a single GPU. This particular container is sophisticated enough to automatically switch between CPU and GPU computation depending on the availability of a GPU device. If you'd like to compare the time it takes for this script to run purely on a CPU, simply omit the `--nv` flag from your call to `apptainer` above and run the command on the same node. You should observe that the runtime jumps to ~252 seconds, meaning that the GPU computation is almost 10 times faster than the CPU!

## Runtime Considerations and Best Practices

This section summarizes key points discussed from the Tensorflow example above and provides extra information regarding best practices.

### Provide the `--nv` flag to Apptainer Runtime

Once you allocate at least one GPU card in your job, you then need to make Apptainer recognize the GPU resources you wish to use. To accomplish this, you can supply the `--nv` flag to an `apptainer shell ...` or `apptainer exec ...` command. Using a generic `gpu_accelerated_tensorflow.sif` image as an example:

```
apptainer exec --nv gpu_accelerated_tensorflow.sif python tensorflow.py
```

### Bind Mounting Directories

By default, most containers only mount your `$HOME` folder, current working directory, and a handful of other common folders. If a host directory isn't in this list and isn't explicitly provided during runtime, you may get a "File not found" error. For example, if you are running a container from `$LOCAL_SCRATCH` and want to write a result file to a `/projects` location, you will need to provide the mount path with the `-B </path/on/host>:</path/in/container>` option:

```
apptainer -B /projects:/projects --nv exec IMAGE.sif COMMAND > /projects/my-project/result.txt
```

### Providing CUDA to Host Environment
In the Tensorflow example above, the container included all of the necessary software to run on a GPU, including CUDA. However, depending on the specific software container you are trying to run, its image may or may not include a working version of CUDA. If you encounter CUDA- or GPU-driver errors, try loading version 12.3 of the CUDA module before running the container:

```
module load cuda/12.3
```

### Change Apptainer cache location to `$LOCAL_SCRATCH`

By default, Apptainer will cache image layers to your `$HOME` folder when you pull or build `.sif` images, which is not ideal as users have a limited storage quota in `/home`. As you continue to use Apptainer, this cache folder can become quite large and can easily fill your `$HOME`. Fortunately, the location of this cache folder can be controlled through the `APPTAINER_CACHEDIR` environmental variable. To avoid overfilling your `$HOME` with unnecessary cached data, it is recommended to add an `APPTAINER_CACHEDIR` location to your `~/.bashrc` file. You can accomplish this with the following command, which will direct these layers to save to a given system's scratch space:

`echo "export APPTAINER_CACHEDIR=$LOCAL_SCRATCH/.apptainer" >> ~/.bashrc`

Note that you will either need to log out and back into the system, or run `source ~/.bashrc` for the above change to take effect.

### Save `.def` files to home folder and images to `$LOCAL_SCRATCH` or `/projects`

An Apptainer definition file (`.def`) is a relatively small text file that contains much (if not all) of the build context for a given image. Since your `$HOME` folders on NREL's HPC systems are regularly backed up, it is strongly recommended to save this file to your home directory in case it accidentally gets deleted or otherwise lost. Since `.sif` images themselves are 1. typically large and 2. can be rebuilt from the `.def` files, we recommend saving them to a folder outside of your `$HOME`, for similar reasons described in the previous section. If you intend to work with an image briefly or intermittantly, it may make sense to save the `.sif` to your `$LOCAL_SCRATCH` folder, from which files can be purged if they haven't been accessed for 28 days. If you plan to use an image frequently over time or share it with other users in your allocation, saving it in a `/projects/` location you have access to may be better.
