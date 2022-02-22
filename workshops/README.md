# NREL HPC Training Workshops 

This section contains links to the finalized presentations and materials we used 
during various user-training workshops. Descriptions of the workshops and any assets 
related to them which have been polished and placed on our website will be provided for each workshop below.

For a schedule and calendar invitations detailing upcoming workshops can be found 
on [our HPC training page](https://www.nrel.gov/hpc/training.html "NREL HPC training"). This page will 
also contain information on recurring/episodic and recently held workshops.

## Recent HPC Workshops (FY22)

### Message Passing Interface Library (MPI): 4 Part Series
<details open>
<summary>More information</summary>
<br>
 
The Message Passing Interface Library (MPI) is used extensively to create parallel applications for high-performance computing systems such as Eagle. It can also be used on small clusters and even on desktop machines. MPI primarily addresses the message-passing parallel programming model, in which data is moved from one process to another process by passing messages.
</details>

## Recent HPC Workshops (FY21)


### Software Environments on Eagle (August 31, 2021)
<details open>
<summary>More information</summary>
<br>
Getting the software or analysis tools you need for your work can be a challenge. This workshop will discuss and demonstrate three common ways of getting your software environment set up on Eagle. Environment modules, Conda, and containers all have associated pros and cons which will be overviewed.

We will provide a background of how each technology works and common challenges. Effectively managing the software you use can greatly reduce the barriers to running your analysis, promote the portability of your work, and in some cases, speed it up!

[Workshop Material](./software_envs)
</details>

### Reinforcement Learning techniques on Eagle (July 22 & 29, 2021)

<details open>
<summary>More information</summary>
<br>

Reinforcement Learning (RL) is a family of key techniques for controlling autonomous systems and other dynamic, stochastic control processes. This 2-part workshop will demonstrate the basics of running RL experiments on Eagle, using OpenAI Gym and RLlib.
 
OpenAI Gym: toolkit that provides a wide variety of simulated environments (Atari games, board games, 2D and 3D physical simulations, and so on), so one can train agents, compare them, or develop new Reinforcement Learning algorithms.
RLlib: open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch.
 
The workshop aims to demonstrate installing Anaconda environments with all necessary packages, allocate Eagle computing resources (CPUs and/or GPUs) for efficiently agent training, and evaluate the results. Time permitting, we will also see how one can write their custom-made Gym environments.
</details>

### Introduction to NREL HPC Workflows (July 13th, 2021)
<details open>
<summary>More information</summary>
<br>
What is a workflow? Why should I use a workflow management system? This presentation will provide a high-level overview of workflow management systems (WMS), and the benefits they can provide to researchers. Features such as automatic parallelization of tasks, portability between computational resources, and increased reproducibility of analysis are just some of the many reasons to use a WMS. Picking a WMS to use can be a daunting task as there are many options to choose from. Several popular WMS will be surveyed and methods for assessing systems that may be a good fit for your work will be discussed. Come and learn how workflows can benefit your research!

[Workshop Material](./intro_to_workflows)
</details>

### Introduction to HPC Concepts (June 30th, 2021)
<details open>
<summary>More information</summary>
<br>
In this session, we will discuss the importance of parallel and high-performance computing. We will, by example, show the basic concepts of parallel computing as well as the advantages and disadvantages of parallel computing. We will present an overview of current and future trends in HPC hardware. We will also provide a very brief overview of some of the paradigms of HPC, including OpenMP, Message Passing Interface (MPI), and GPU programming.
</details>

### Introduction to NREL HPC Systems (June 24th, 2021)
<details open>
<summary>More information</summary>
<br>
Dive in to NREL's prominent HPC system, Eagle. Learn about system configuration, storage systems, submitting workloads, and helpful hints and tips!

Explore NREL HPC File Systems
Software environments and modules
Introduction to Slurm
Becoming productive with Eagle
</details>

### Linux Fundamentals: Utilizing the Command Line Interface (June 16th, 2021)
<details open>
<summary>More information</summary>
<br>
In this introduction to linux, we will demonstrate the fundamental commands and concepts required to make use of NREL HPC linux systems (or any *nix operating system.) We will cover:
Setting up your computer to utilize HPC linux systems
Using the Linux command line
Remote system access (SSH)
Linux file permissions basics
</details>

### Advanced Jupyter (May 13th, 2021)
<details open>
<summary>More information</summary>
<br>
Beyond the basics: this advanced Jupyter workshop will survey topics which enable you to get more out of your interactive notebooks. It will build on the recent Intro to Jupyter workshop and introduce additional Magic commands. Interacting with Slurm from a notebook will also be covered, and how this can be used to achieve multi-node parallelism. Additional topics include utilizing GPUs from a notebook, and parameterized notebook execution with Papermill.
</details>

<br>

### TensorFlow on Eagle (April, 26 2021)
<details open>
<summary>More information</summary>
<br>
In this introduction to using TensorFlow on Eagle, we will discuss what TensorFlow is, why you may want to use TensorFlow, and how you may install and use pre-compiled versions of TensorFlow which are specifically optimized for Eagle’s CPU and GPU architecture. This workshop includes instructions for accessing and installing the optimized versions of TensorFlow for Eagle, as well as some basic steps for testing your TensorFlow installation and performing benchmarking tests.
</details>

<br>

### Using the Slurm Job Scheduler (April 8th, 2021)

<details open>
<summary>More information</summary>
<br>
Slurm is the batch queuing system for running jobs on Eagle and many other HPC systems. The heart of running under slurm is a slurm script. After a quick review of slurm basics we will dive into a number of slurm example scripts, highlighting methods of getting good utilization of HPC resources. Each example was originally written to address a particular question such as: “How do I do X in slurm?” Some of the topics to be covered include:

- effectively mapping tasks and threads to cores
- creating scripts to promote reproducible results
- running with different MPI executables on various cores
- getting inputs from your environment to enable submitting multiple jobs with different inputs without changing scripts
- creating job dependencies and job arrays
- running both a CPU and GPU job in a single script and forcing task affinity.  

Content on this workshop can be found in the [slurm examples slides](slurm_examples_slides) directory of this repository.
Additional content and examples can be found in [Tim Kaiser's Github repository](https://github.com/timkphd/examples/tree/master/tims_tools).

</details>




### Introduction to Jupyter on NREL HPC Resources (March 30, 2021)

<details open>
<summary>More information</summary>
<br>
In this introduction to Jupyter, we will discuss what Jupyter is, why you may 
want to use Jupyter, and how you may utilize notebooks on the various available 
NREL compute resources. This workshop includes a live demonstration with methods 
for using JupyterHub on Eagle (Europa), Jupyter notebooks on Eagle nodes, as well as 
sample notebooks and cluster submission scripts.

Content for this presentation can be found in the [jupyter_intro_slides](jupyter_intro_slides) directory.


</details>

<br>

## Upcoming HPC Workshops




<br>

---

<details>

<summary> Older Workshops </summary>
<br>

### Workshop - Slurm: Advanced Techniques (held March 20th, 2019)
<details>
<summary>More information</summary>
<br>

The second of our series, Eagle Workshop - Advanced Slurm Techniques, covered topics beneficial for job management:

* Job monitoring and forensics: usage examples on sreport, sacct, sinfo, and sview (FastX)
* Advanced srun and sbatch functions (flags)
* Parallelizing with SLURM
* Remote exclusive GPU usage, requesting GPU nodes.

The resources used during this presentation are available here:

* [Presentation Slides](https://www.nrel.gov/hpc/assets/pdfs/slurm-advanced-topics.pdf)

</details>


### Workshop - Slurm: New NREL Capabilities (held March 8th, 2019)

<details>
<summary>More information</summary>
<br>

This workshop covered the following features which are new to the NREL HPC workflow relative to what was possible on Peregrine and its job scheduler:

*   Basic Slurm core functionality overview
*   Slurm partitions - request by features
    *   Effective queue partition requests
    *   Request by resource needs
        *   GPU compute nodes
        *   Local scratch
        *   Memory requirements
*   Job dependencies and job arrays
*   Job steps
*   Job monitoring and basic troubleshooting. 

The resources used during this presentation are available here:

* [New Features Offered by Slurm - Presentation Slides](https://www.nrel.gov/hpc/assets/pdfs/slurm-new-nrel-capabilities-presentation.pdf)

</details>

### Transition from Peregrine to Eagle (held January 11th, 2019)

<details>
<summary>More information</summary>
<br>

The HPC Operations team held workshops for providing live assistance with acclimating to Eagle, and is developing similar sessions to help users get the most out of HPC resources. The resources used during these presentations are available here:

* [Transitioning from Peregrine to Eagle - Presentation Slides](https://www.nrel.gov/hpc/assets/pdfs/peregrine-to-eagle-transition-presentation.pdf "Peregrine to Eagle Presentation Slides")
* [Separate instructions for how to use Globus to migrate files quickly and reliably](https://www.nrel.gov/hpc/assets/pdfs/using-globus-to-move-data-from-peregrine-to-eagle.pdf)
* [PBS to Slurm Analogous Command Cheat Sheet](https://www.nrel.gov/hpc/assets/pdfs/pbs-to-slurm-translation-sheet.pdf)

</details>

</details>

