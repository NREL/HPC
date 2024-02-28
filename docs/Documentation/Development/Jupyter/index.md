
# Introduction to Jupyter

## What is Jupyter?

**A web app for interactive Python in a browser** 

- "Live coding"
- Instant visualization
- Sharable
- Reproducible
- Customizable
- Now supports other languages besides Python (R, Julia..)   
    - https://github.com/jupyter/jupyter/wiki/Jupyter-kernels


## Example Notebook Code

```python
import chart_studio.plotly as py
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
x = np.linspace(0, 5, 10)
y = x ** 2
n = np.array([0,1,2,3,4,5])
xx = np.linspace(-0.75, 1., 100)

fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].scatter(xx, xx + 1.25*np.random.randn(len(xx)))
#axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")

axes[1].step(n, n**2.0, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2.5, x**3, color="green", alpha=0.5);
axes[3].set_title("fill_between");
```


![png](../../../../assets/images/output_4_0.png)


## Jupyter Terminology


### **Jupyterhub**
    * Multi-user "backend" server
    * Controls launching the single-user Jupyter server
    * NREL's "Europa" (Eagle-only) runs Jupyterhub
 
(In general, don't worry about JupyterHub--unless you're a sysadmin)

### **Jupyter/Jupyter Server/Notebook server**
    * The single-user server/web interface
    * Create/save/load .ipynb notebook files
    * What users generally interact with

### **Jupyter Notebook**
    * An individual .pynb file
    * Contains your Python code and visualizations
    * Sharable/downloadable

###  **Jupyter lab**
    * A "nicer" web interface for Jupyter - "notebooks 2.0"
    * Preferred by some
    * Lacking some features of "classic" notebooks

### **Kernel**
    * The Python environment used by a notebook
    * More on kernels later

## Using Europa

The NREL HPC team runs a Jupyterhub server, called Europa, that is available for internal (NREL) Eagle users only. 

Europa is connected to Eagle's Lustre storage system for access to /projects data.

A replacement for Europa on Kestrel is in the planning stage.

### Europa's Advantages:
    * Fast and easy access
    * Use regular Eagle credentials
    * Great for light to moderate processing/debugging/testing

### Europa's Disadvantages:
    * Limited resource: 8 cores/128GB RAM per user beefore automatic throttling
    * Compete with other users for CPU/RAM on a single machine
    * Not available for Kestrel (yet)

### Simple Instructions:
    
    - Visit Europa at (https://europa.hpc.nrel.gov/)
    
    - Log in using your HPC credentials
     
    - Opens a standard "notebooks" interface
     
    - Change url end /tree to /lab for Lab interface


## Using a Compute Node

### Advantages:
    * Custom environments
    * 36 cores and up to ~750GB RAM
    * No competing with other users for cores

### Disadvantages:
    * Compete with other users for nodes
    * Costs AU
    
## Launching Your Own Jupyter Server on an HPC System

Both Kestrel and Eagle support running your own Jupyter Notebook server.

External (non-NREL) **Kestrel** users may follow the directions below for Kestrel, but please use `kestrel.nrel.gov` instead of `kestrel.hpc.nrel.gov`. 

External (non-NREL) **Eagle** users will no longer be able to use Jupyter in this fashion as of February 2024. If you require Jupyter, please consider transitioning to Kestrel as soon as possible.

## Using a Compute Node to run Jupyter Notebooks

Connect to a login node and request an interactive job using the `salloc` command.

The examples below will start a 2-hour job. Edit the `<account>` to the name of your allocation, and adjust the time accordingly. Since these are interactive jobs, they will get some priority, especially if they're shorter, so only book as much time as you will be actively working on the notebook.

Before you get started, we recommend installing your own Jupyter inside of a conda environment. The default conda/anaconda3 modules contain basic Jupyter Notebook servers, but you will likely want your own Python libraries, notebook extensions, and other features. Basic directions are included later in this document.

### Kestrel:

`[user@laptop:~]$ ssh kestrel.hpc.nrel.gov`

`[user@kl1:~]$ salloc -A <account> -t 02:00:00`

### Eagle:

`[user@laptop:~]$ ssh eagle.hpc.nrel.gov`

`[user@el1:~]$ salloc -A <account> -t 02:00:00`


## Starting Jupyter Inside the Job

Once the job starts and you are allocated a compute node, load the appropriate modules, activate your Jupyter environment, and launch the Jupyter server.

#### Kestrel:

`[user@x1000c0s0b0n1:~]$ module load anaconda3`

`[user@x1000c0s0b0n1:~]$ source activate myjupenv`

`[user@x1000c0s0b0n1:~]$ jupyter-notebook --no-browser --ip=$(hostname -s)`

Take note of the node name that your job is assigned. (x1000c0s0b0n1 in the above example.)

Also note the url that Jupyter displays when starting up, e.g. `http://127.0.0.1:8888/?token=<alphabet soup>`.

The `<alphabet soup>` is a long string of letters and numbers. This is a unique authorization token for your Jupyter session. you will need it, along with the full URL, for a later step.

#### Eagle:

`[user@r2i7n35:~]$ module load conda`

`source activate myjupenv`

`jupyter-notebook --no-browser --ip=$(hostname -s)`

Take note of the node name that your job is assigned. (r2i7n35 in this example.)

Also note the url that Jupyter displays when starting up, e.g. `http://127.0.0.1:8888/?token=<alphabet soup>`.

The `<alphabet soup>` is a long string of letters and numbers. This is a unique authorization token for your Jupyter session. you will need it, along with the full URL, for a later step.

### On Your Own Computer

Next, open an SSH tunnel through a login node to the compute node. Log in when prompted using your regular HPC credentials, and put this terminal to the side or minimize it, but leave it open until you are done working with Jupyter for this session.

#### Kestrel: 

`[user@laptop:~]$ ssh -N -L 8888:<nodename>:8888 username@eagle.hpc.nrel.gov`


#### Eagle:

`[user@laptop:~]$ ssh -N -L 8888:<nodename>:8888 username@eagle.hpc.nrel.gov`


### Open a Web Browser

Copy the full url and token from Jupyter startup into your web browser. For example:

`http://127.0.0.1:8888/?token=<alphabet soup>`


## Using a Compute Node on Eagle - Easy Way

Scripted assistance with launching a Jupyter session on Eagle is available.

These scripts are designed for Eagle and have not yet been adapted for Kestrel, but may be downloaded and adapted manually.

### pyeagle - NREL Users

The [pyeagle](https://github.nrel.gov/MBAP/pyeagle) package is available for internal users to handle launching and monitoring a jupyter server on a compute node. This package is maintained by an NREL HPC user group, and provides utilities for working on Eagle and Kestrel.

###  Auto-launching on Eagle With an sbatch Script

Full directions included in the [Jupyter repo](https://github.com/NREL/HPC/tree/master/general/Jupyterhub/jupyter).

Download [sbatch_jupyter.sh](https://github.com/NREL/HPC/blob/master/general/Jupyterhub/jupyter/sbatch_jupyter.sh) and [auto_launch_jupyter.sh](https://github.com/NREL/HPC/blob/master/general/Jupyterhub/jupyter/auto_launch_jupyter.sh)

Edit [sbatch_jupyter.sh](https://github.com/NREL/HPC/blob/master/general/Jupyterhub/jupyter/sbatch_jupyter.sh) to change:

`--account=*yourallocation*`

`--time=*timelimit*`

Run [auto_launch_jupyter.sh](https://github.com/NREL/HPC/blob/master/general/Jupyterhub/jupyter/auto_launch_jupyter.sh) and follow directions

That's it!

## Using a Login Node

Please avoid running Jupyter on a login node on either Kestrel or Eagle. 

### Reasons to Not Run Jupyter Directly on a Login Node

    * Heavy lifting should be done via Europa or compute nodes
    * Using a highly shared resource (login nodes)
        * Competition for cycles
        * arbiter2 will throttle moderate to heavy usage

## Custom Conda Environments and Jupyter Kernels

On Kestrel, the module 'anaconda3' is available for the conda environment manager. As an alternative, the module 'mamba' is available as well. Mamba is conda-compatible and has a faster solver.

On Eagle, the module 'conda' contains the conda environment manager. The Eagle conda module also contains mamba installed as a conda package.

### Creating a Conda Environment

Create an environment and install the base jupyter packages, then activate the environment and install other libraries, e.g. scipy, numpy, and so on.

`conda create -n myjupyter -c conda-forge jupyter ipykernel`

`source activate myjupyter`

`conda install -c conda-forge scipy numpy matplotlib`

### Add Custom iPykernel

`python -m ipykernel install --user --name=myjupyter`

If you already have a Jupyter server running, restart it to load the new kernel.

The new kernel will appear in the drop-down as an option to open a new notebook.

### Remove Custom iPykernel

`jupyter kernelspec list`

`jupyter kernelspec remove myoldjupyter`


## Magic Commands

Magic commands are "meta commands" that add extra functionality.

Magic commands begin with % or %%.

### A Few Useful Examples
 
    * %lsmagic - list all magic commands
    * %run _file.py_ - run an external python script
    * %%time - placed at top of cell, prints execution time
    * %who - list all defined variables in notebook
    
```
%lsmagic

```

    Available line magics:
    %alias  %alias_magic  %autoawait  %autocall  %automagic  %autosave  %bookmark  %cat  %cd  %clear  %colors  %conda  %config  %connect_info  %cp  %debug  %dhist  %dirs  %doctest_mode  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %lf  %lk  %ll  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %lx  %macro  %magic  %man  %matplotlib  %mkdir  %more  %mv  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %pip  %popd  %pprint  %precision  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %rep  %rerun  %reset  %reset_selective  %rm  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%markdown  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.

## Shell Commands

You can also run shell commands inside a cell. For example:

`!conda list` - see the packages installed in the environment you're using


```python
!pwd
!ls
```

    /home/tthatche/jup
    auto_launch_jupyter.sh	  Jupyter Presentation.ipynb  slurm-6445885.out
    geojsondemo.ipynb	      old			              sshot1.png
    Interesting Graphs.ipynb  sbatch_jupyter.sh	          sshot2.png
    jup-logo.png		      slurm


## Interesting/Useful Notebooks

[Awesome Jupyter](https://github.com/markusschanta/awesome-jupyter)

[Awesome Jupyterlab](https://github.com/mauhai/awesome-jupyterlab)

[Plotting with matplotlib](https://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb)

[Python for Data Science](https://nbviewer.jupyter.org/github/gumption/Python_for_Data_Science/blob/master/Python_for_Data_Science_all.ipynb)

[Numerical Computing in Python](https://nbviewer.jupyter.org/github/phelps-sg/python-bigdata/blob/master/src/main/ipynb/numerical-slides.ipynb)

[The Sound of Hydrogen](https://nbviewer.jupyter.org/github/Carreau/posts/blob/master/07-the-sound-of-hydrogen.ipynb)

[Plotting Pitfalls](https://anaconda.org/jbednar/plotting_pitfalls/notebook)

[GeoJSON Extension](https://github.com/jupyterlab/jupyter-renderers/tree/master/packages/geojson-extension)


## Happy Notebooking!

