## Installing PyTorch on Kestrel

To install PyTorch for use with GPUs on Kestrel, the first step is to load the anaconda module on the GPU node using ```module load conda```. Once the anaconda module has been loaded, create a new environment in which to install PyTorch, e.g.,

??? example "Creating and activating a new conda environment"
        conda create --prefix /projects/<your-project-name>/<your-user-name>/<conda-env-dir>/pt python=3.9
        conda activate /projects/<your-project-name>/<your-user-name>/<conda-env-dir>/pt

!!! Note
	If you are not familiar with using [Anaconda environments](https://www.anaconda.com/) please refer to the [NREL HPC page on using Conda environments](../../Documentation/Environment/Customization/conda.md) and also the Conda guide to [managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Once the environment has been activated, you can install PyTorch using the standard approach found under the Get Started tab of the [PyTorch](https://pytorch.org/) website, e.g., using ```pip```,

??? example "Installing PyTorch using pip"
	```pip3 install torch torchvision torchaudio```

!!! Note
	Currently, we recommend installing software using the GPU nodes.

## Running a PyTorch Batch Job on Kestrel

??? example "Sample job script: Kestrel - Shared (partial) GPU node"

    ```
    #!/bin/bash
    #SBATCH --account=<your-account-name> 
    #SBATCH --reservation=<friendly-users-reservation>
    #SBATCH --partition=gpu-h100
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:h100:1 
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=1
    #SBATCH --time=02:00:00
    #SBATCH --job-name=<your-job-name>

    module load conda
    conda activate /projects/<your-project-name>/<your-user-name>/<conda-env-dir>/pt

    srun python <your-pytorch-code>.py
    ```

## PyTorch Example
Below we present a simple convolutional neual network example for getting started using PyTorch with Kestrel GPUs. The original, more detailed version of this example can be found in the pytorch tutorials repo [here](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py).

??? example "CIFAR10 example"
    ```
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    # Check if there are GPUs. If so, use the first one in the list
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load data and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    # Define the CNN
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # send the network to the device
    # If you want to use data parallelism across multiple GPUs, uncomment if statement below
    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net)
    
    net.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data # setup without device
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    ```

!!! Note
	Currently, this code will run on a single GPU, specifically the GPU denoted ```cuda:0```. To use multiple GPUs via data parallelism, uncomment the two lines above the ```net.to(device)``` command. Furthermore, use of multiple GPUs require requesting multiple GPUs for the batch or interactive job.

!!! Note
	To better observe the multi-GPU peformance of the above example, you can change the size of the CNN. For example, by increasing the size of the second argument in the definition of ```self.conv1``` and the first argument in ```self.conv2```, you can increase the size of the network and use more resources for training.
