# Using ParaView in Client-Server Mode

Running ParaView interactively in client-server mode is a convenient worflow for researchers who have a large amount of remotely-stored data that they'd like to visualize using a locally-installed copy of ParaView.  In this model, the Eagle HPC does the heavy lifting of reading file data and applying filters, taking advantage of parallel processing when possible, then "serves" the rendered data to the ParaView client running locally on your desktop.  This allows you to interact with ParaView as you normally would (i.e., locally) with all your preferences and shortcuts intact *without* the time consuming step of transferring data from Eagle to your desktop or relying on a remote desktop environment.

This guide assumes that you already have a copy of the ParaView software installed on your local computer.  If not, you can [download a copy here](https://www.paraview.org/download/).  For the most seamless client-server experience, it's best to match your desktop version of ParaView to the version installed on the cluster.  To determine which version of ParaView is installed on the cluster, connect to Eagle as you normally would, load the ParaView module with `module load paraview`, then check the version with `pvserver --version`.  The version number, e.g., 5.6.0, will then be displayed to your terminal.  If your local copy of ParaView matches the major version number, e.g., 5.x, it may be compatible, but exact matching ensures success.

For Windows users, it is additionally necessary to install the PuTTY software to complete the steps carried out in the terminal.  You can [download it here](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html).

### 1. Reserve Compute Nodes
The first step is to reserve the computational resources on Eagle that will be running the ParaView server. 

This requires using the Slurm `salloc` directive and specifying an allocation name and time limit for the reservation. (This is one of the rare times where `salloc` is used instead of srun.)

```bash
salloc -A <alloc_name> -t <time_limit> 
```

where `<alloc_name>` will be replaced with the allocation name you wish to charge your time to and `<time_limit>` is the amount of time you're reserving the nodes for.  At this point, you may want to copy the name of the node that the Slurm scheduler assigns you (it will look something like r1i0n10, r4i3n3, etc., and follow immediately after the "@" symbol at the command prompt ) as we'll need it in Step 3.

In the example above, we default to requesting only a single node which limits the maximum number of ParaView server processes we can launch to the maximum number of cores on a single Eagle node (on Eagle, this is 36).  If you intend to launch more ParaView server processes than this, you'll need to request multiple nodes with your `salloc` command.

```bash
salloc -A <alloc_name> -t <time_limit> -N 2
```

where the `-N 2` option specifies that two nodes be reserved, which means the maximum number of ParaView servers that can be launched in Step 2 is 36 x 2 = 72.  Although this means you'll be granted multiple nodes with multiple names, the one to copy for Step 3 is still the one immediately following the "@" symbol.  See the table of recommended workload distributions in Step 2 for more insight regarding the number of nodes to request.

### 2. Start ParaView Server
After reserving the compute nodes, load the ParaView module with

```bash
module load paraview
```

Next, start the ParaView server with another call to the Slrum `srun` directive

```bash
srun -n 8 pvserver --force-offscreen-rendering
```

In this example, the ParaView server will be started on 8 processes.  The `--force-offscreen-rendering` option is present to ensure that, where possible, CPU-intensive filters and rendering calculations will be performed server-side (i.e., on the Eagle compute nodes) and *not* on your local machine.  Remember that the maximum number of ParaView server processes that can be launched is limited by the amount of nodes reserved in Step 1.  Although every dataset may be different, ParaView offers the following recommendations for balancing grid cells to processors.

| Grid Type         | Target Cells/Process | Max Cells/Process |
| ----------------- | -------------------- | ----------------- |
| Structured Data   | 5-10 M               | 20 M              |
| Unstructured Data | 250-500 K            | 1 M               |

So for example, if you have data stored in an unstructured mesh with 6 M cells, you'd want to aim for between 12 and 24 ParaView server processes, which easily fits on a single Eagle node.  If the number of unstructured mesh cells was instead around 60 M, you'd want to aim for 120 to 240 processes, which means requesting a minimum of 4 eagle nodes at the low end (36 x 4 = 144).  Note, this 4-node request may remain in the queue longer while the scheduler looks for resources, so depending on your needs, it may be necessary to factor queue times into your optimal cells-per-process calculation.

Note: The `--server-port=<port>` option may be used with pvserver if you wish to use a port other than 11111 for Paraview. You'll need to adjust the port in the SSH tunnel and tell your Paraview client which port to use, as well. See the following sections for details.



### 3. Create SSH Tunnel
Next, we'll create what's called an SSH tunnel to connect your local desktop to the compute node(s) you reserved in Step 1.  This will allow your local installation of ParaView to interact with files stored remotely on Eagle.  **In a new terminal window**, execute the following line of code **on your own computer**:

```bash
ssh -L 11111:<node_name>:11111 <user_name>@eagle.hpc.nrel.gov
```

where `<node_name>` is the node name you copied in Step 1 and `<user_name>` is your HPC username. 

Note that if you changed the default port to something other than 11111 (see the previous section) you'll need to change the port settings in your SSH tunnel, as well. The SSH command construct above follows the format of `<local_port>:<node_name>:<remote_port>`. The `<local_port>` is the "beginning" of the tunnel on your computer, and is often the same as the "end" port of the tunnel, though this is not required. You may set this to anything convenient to you, but you will need to tell your Paraview client the right port if you change it (see the next section for details.) <remote_port> is the port on the Eagle compute node where pvserver is running. The default for pvserver is 11111, but if you changed this with pvserver `--server-port=` flag, you'll need to change <remote_port> in your ssh command to match.

### 4. Connect ParaView Client
Now that the ParaView server is running on a compute node and your desktop is connected via the SSH tunnel, you can open ParaView as usual.  From here, click the "Connect" icon or `File > Connect`.  Next, click the "Add Server" button and enter the following information.

| Name        | Value         |
|-------------|---------------|
| Name        | Eagle HPC     |
| Server Type | Client/Server |
| Host        | localhost     |
| Port        | 11111         |

Only the last three fields, Server Type, Host, and Port, are strictly necessary (and many of them will appear by default) while the Name field can be any recognizable string you wish to associate with this connection.  When these 4 fields have been entered, click "Configure" to move to the next screen, where we'll leave the Startup Type set to "Manual".  Note that these setup steps only need to be completed the first time you connect to the ParaView server, future post-processing sessions will require only that you double click on this saved connection to launch it.

When finished, select the server just created and click "Connect".  The simplest way to confirm that the ParaView server is running as expected is to view the Memory Inspector toolbar (`View > Memory Inspector`) where you should see a ParaView server for each process started in Step 2 (e.g., if `-n 8` was specified, processes `0-7` should be visible).

That's it!  You can now `File > Open` your data files as you normally would, but instead of your local hard drive you'll be presented with a list of the files stored on Eagle.

### General Tips
* The amount of time you can spend in a post-processing session is limited by the time limit specified when reserving the compute nodes in Step 1.  If saving a large time series to a video file, your reservation time may expire before the video is finished.  Keep this in mind and make sure you reserve the nodes long enough to complete your job.
* Adding more parallel processes in Step 2, e.g., `-n 36`, doesn't necessarily mean you'll be splitting the data into 36 blocks for each operation.  ParaView has the *capability* to use 36 parallel processes, but may use many fewer as it calculates the right balance between computational power and the additional overhead of communication between processors.

