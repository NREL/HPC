# NREL HPC Training Workshops 

This section contains links to the finalized presentations and materials we used 
during various user-training workshops. Descriptions of the workshops and any assets 
related to them which have been polished and placed on our website will be provided for each workshop below.

For a schedule and calendar invitations detailing upcoming workshops can be found 
on [our HPC training page](https://www.nrel.gov/hpc/training.html "NREL HPC training"). This page will 
also contain information on recurring/episodic and recently held workshops.


## Recent HPC Workshops (FY21)


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

