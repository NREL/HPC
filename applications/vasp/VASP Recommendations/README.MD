A study was performed to evaluate the performance on VASP on Swift and Eagle using [ESIF VASP Benchmarks](https://github.com/NREL/ESIFHPC3/tree/master/VASP) 1 and 2. Benchmark 1 is a system of 16 atoms (Cu4In4Se8), and Benchmark 2 is a system of 519 atoms (Ag504C4H10S1).

On Swift, the default builds of VASP installed on the system as modules were used. The Intel MPI build was built with Intel compilers and the mkl math library, and it was accessed via the "vaspintel" module. The OpenMPI build was compiled with gnu using gcc and fortran compilers and used OpenMPI's math libraries, and was accessed via the "vasp" module. Both builds run VASP 6.1.1.

On Eagle, the default build of VASP installed on the system is an Intel MPI version of VASP. The Intel MPI build was built with Intel compilers and the mkl math library, and it was accessed via the "vasp" module. It runs VASP 6.1.2. No Open MPI VASP build is accessible through the default modules on Eagle, but an Open MPI build can be accessed in an environment via "source /nopt/nrel/apps/210830a/myenv.2108301742, ml vasp/6.1.1-l2mkbb2". The OpenMPI build was compiled with gnu using gcc and fortran compilers and used OpenMPI's math libraries. It runs VASP 6.1.1.

The [VASP repo](https://github.com/claralarson/HPC/tree/master/applications/vasp/VASP%20Recommendations) contains scripts that can be used to run the Intel MPI and Open MPI builds used in the study to perform calculations on Swift and Eagle. Additionally, the repo contains the VASP peroformance data collected, a notebook that analysis the performance data, data on CPU usage over time with different values of cpu-bind, and a notebook that analyses and visualizes the cpu-bind data. 

This document synthesises the performance data analysis in order to provide information about how to most efficiently run VASP for users running VASP on Swift and Eagle. 

## Eagle

### Recommended CPUs/Node

Run VASP on full nodes (36 CPUs/node). While using fewer cores per node yields an improvement in runtime/core, it will results in a larger allocation charge, as Eagle charges per node used, regardless of how much of the node is used. Eagle charges 3AUs/node-hour. 

### VASP on GPU Nodes

Two versions of the vasp_gpu executable are available on Eagle. The older vasp_gpu build on Eagle is built using CUDA, and the newer version is built using OpenACC. The OpenACC GPU build was found to run in an average of 27.7% of the time as the old GPU build using Benchmark 2. Little runtime improvement (compared to running on CPUs) was seen using the older, CUDA GPU build. Find a script for running the OpenACC GPU build in [this section](#Scripts-for-Running-VASP-on-Eagle).

Running the OpenACC GPU build of VASP (vasp_gpu) on GPU nodes improves performance for larger VASP calculations, but may increase the runtime for smaller calculations. Benchmark 2 calculations on using the OpenACC build on GPU nodes ran in an average of 27.7% of the time as CPU calculations on the same number of nodes, but Benchmark 1 calculationson using the OpenACC build on GPU nodes ran in an average of 150% of the time as CPU calculations on the same number of nodes.

   * Memory limitation: GPU nodes on Eagle cannot provide as much memory as CPU nodes for VASP jobs, and large VASP jobs may require more GPU nodes to provide enough memory for the calculation. For Benchmark 2, at least 2 full nodes were needed to provide enough memory to complete a calculation. Using more complicated parallelization schemes, the number of nodes necessary to provide enough memory scaled with the increase in number of problems handled simultaneousely. 

![Eagle GPU Bench 2](https://github.com/claralarson/HPC/blob/4837ef43a7d03b1b34a1e5ceda8dcbc64ca5b128/applications/vasp/VASP%20Recommendations/Images/Eagle_GPU_2.png)

![Eagle GPU Bench 1 4x4x2](https://github.com/claralarson/HPC/blob/4837ef43a7d03b1b34a1e5ceda8dcbc64ca5b128/applications/vasp/VASP%20Recommendations/Images/Eagle_GPU_1_4x4x2.png)

### MPI

Intel MPI is recommended over Open MPI. Using an Intel MPI build of VASP and running over Intel MPI, Benchmark 2 ran in average of 50% of the time as the same calculations using an Open MPI build of VASP over Open MPI. For Benchmark 1, Intel MPI calculations ran in an average of 63.5% of the time as Open MPI calculcations. 

Find scripts for running the Intel MPI and Open MPI builds of VASP in [this section](#Scripts-for-Running-VASP-on-Eagle).

### --cpu-bind Flag
The --cpu-bind flag changes how tasks are assigned to cores throughout the node. Setting --cpu-bind=cores or rank showed no improvement in the performance of VASP on 36 CPUs/node. Using 18 CPUs/node setting --cpu-bind=cores shows a small improvement in runtime (~5% decrease) using both Intel MPI and Open MPI. (See [LINK TO CPU-BIND DOC] for info on the effect of cpu-bind)

cpu-bind can be set as a flag in an srun command, such as 
```
srun --cpu-bind=cores vasp_std
```
### KPAR

KPAR determines the number of groups across which to divide calculations at each kpoint, and one calculation from each group is performed at a time. The value of KPAR can be defined in the INCAR file. 
  * Per [VASP documentation](https://www.vasp.at/wiki/index.php/KPAR), the KPAR tag file should always be set to a value that evenly divides the total number of cores used. 
  * We found that runtime starts to increase (in other words, runtime scales positively with the number of nodes, rather than negatively) if you increase the number of nodes past the value of KPAR, so it is recommended to set KPAR no lower than the number of nodes used. 
  * Lower values of KPAR might be better for lower node counts. For example, Benchmark 1 calculations on 1-4 nodes were fastest using KPAR=4, but Benchmark 1 calculations on more than 4 nodes were fastest using KPAR=9.
  * The [GPU VASP documentation](https://www.vasp.at/wiki/index.php/OpenACC_GPU_port_of_VASP) recommends setting KPAR equal to the total number of GPUs used. Our results are relatively consistent with this recommendation. 

### K-Points Scaling

Runtime does not scale well with the number of kpoints. Benchmark 1 uses a 10x10x5 kpoints grid (500 kpoints). When run with a 4x4x2 kpoints grid (16 kpoints), we should expect the runtime to scale by 16/500 (3.2%) since calculations are being performed at 16 points rather than 500. However, the average scaling factor between Benchmark 1 jobs on Eagle with 10x10x5 grids and 4x4x2 grids is 28% (ranging from ~20%-57%). 

### Scripts for Running VASP on Eagle
  * [VASP on Eagle with Intel MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Eagle_IntelMPI.slurm)
  * [VASP on Eagle with Open MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Eagle_OpenMPI.slurm)
  * [VASP on Eagle on GPUs with OpenACC GPU build using Intel MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Eagle_OpenACC_GPU.slurm)

## Swift

### Recommended CPUs/Node

On Swift, VASP is most efficiently run on partially full nodes. 32 CPUs/node was found to have the fastest runtime/core, followed by 64 CPUs/node and 128 CPUs/node. Compared to jobs on 64 CPUs/node, jobs on 32 CPUs/node using the same total number of cores ran in 70%-90% of the 64 CPUs/node runtime. On Swift, each node has 64 physical cores, and each core is subdivided into two virtual cores in a processes that is identical to hyperthreading. Because of this, up 128 cores can be requested from a single Swift node, but each core will only represent half of a physical core. 

Unlike on Eagle, Swift charges for only the portion of the node requested by a job, as long as the memory requested for the job is no more than 2GB/CPU. If the entire 256GB of memory is requested per node, but only half of the CPUs per node are requested, you will be charged for the full node. Swift charges 5 AU/hour when running on 128 nodes (one full node), so running on 32 CPUs, for example, would charge only (32/128) * 5 AUs/hour rather than the full 5 AUs/node-hour. 

Unlike on Eagle, multiple jobs can run on the same nodes on Swift. This runtime performance was simualted by the "shared" nodes in the graphs. (Find scripts for running multiple VASP jobs on the same nodes in [this section](#Scripts-for-Running-VASP-on-Swift)). So if you are only using a fraction of a node, other users' jobs could be assigned to the rest of the node, which we suspect might deteriorate the performance since "shared" nodes in the graphs below are shown to have the slowest rates. Setting "#SBATCH --exclusive" in your run script prevents other users from using the same node as you, but you will be charged the full 5AUs/node, regardless of the number of CPUs/node you are using. In some cases, running on 32 CPUs/node with the --exculsive flag set might minimize your allocation charge. For example, in the "Open MPI, performance/node" graph, we see that 32 CPUs/node shows consistently the fastest runtime per node for all jobs using 2 or more nodes, so using 32 CPUs/node on 2 nodes could complete faster than 64 CPUs/node on 2 nodes. 

The graphs below are meant to help users identify the number of CPUs/node will be most efficient in running their jobs. The graphs that show "performance/core" match jobs that use the same number of cores (but different number of nodes depending on CPUs/node), and they show the efficiency per core. These graphs are most useful if you will be charged by CPU (i.e. --exclusive tag is not set). The graphs that show "performance/node" match jobs that use the same number of nodes (but different number of cores depending on CPUs/node), and they show efficiency per node. These graphs are most useful if you will be charged by node (i.e. --exclusive tag is set). 

Intel MPI, performance/core  |  Intel MPI, performance/node
:-------------------------:|:-------------------------:
![](https://github.com/claralarson/HPC/blob/b667dfce623fedc54f1d9da5ea28cd59cc346beb/applications/vasp/VASP%20Recommendations/Images/Swift_2_Intel_Cores.png) |  ![](https://github.com/claralarson/HPC/blob/b667dfce623fedc54f1d9da5ea28cd59cc346beb/applications/vasp/VASP%20Recommendations/Images/Swift_2_Intel_Nodes.png)

Open MPI, performance/core  |  Open MPI, performance/node 
:-------------------------:|:-------------------------:
![](https://github.com/claralarson/HPC/blob/ffaa2609e6c5d9b055aae03ba0abcc5c9bc1f7a2/applications/vasp/VASP%20Recommendations/Images/Swift_2_Open_Cores.png)  |  ![](https://github.com/claralarson/HPC/blob/b667dfce623fedc54f1d9da5ea28cd59cc346beb/applications/vasp/VASP%20Recommendations/Images/Swift_2_Open_Nodes.png)

### MPI

Intel MPI is recommended over Open MPI for all VASP calculations on Swift. Using an Intel MPI build of VASP and running over Intel MPI, Benchmark 2 ran in average of 76%, 72% and 46% of the time as the same calculations using an Open MPI build of VASP over Open MPI on 32, 64 and 128 CPUs/node, respectively. For Benchmark 1, Intel MPI calculations ran in an average of CHECK BACK of the time as Open MPI calculcations. 

Find scripts for running the Intel MPI and Open MPI builds of VASP in [this section](#Scripts-for-Running-VASP-on-Swift).

### --cpu-bind Flag

The --cpu-bind flag changes how tasks are assigned to cores throughout the node. On Swift, it is recommended not to use cpu-bind. Running VASP on 64 CPUs/node and 128 CPUs/node, setting --cpu-bind=cores or rank showed no improvement in runtime. Running VASP on 32 CPUs/node, setting --cpu-bind=cores or rank increased runtime by up to 40%. (See [LINK TO CPU-BIND DOC] for info on the effect of cpu-bind)

```
srun --cpu-bind=cores vasp_std
```

### KPAR

KPAR determines the number of groups across which to divide calculations at each kpoint, and one calculation from each group is performed at a time. The value of KPAR can be defined in the INCAR file. 
  * Per [VASP documentation](https://www.vasp.at/wiki/index.php/KPAR), the KPAR tag file should always be set to a value that evenly divides the total number of cores used as well as the total number of KPOINTS. VASP will not run if KPAR does not evenly divide the total number of cores, and so core counts were slightly altered for the jobs represented in the KPAR=9 graph in order to get VASP to run on Swift with KPAR=9. 
  * We found significant difference in runtime for Benchmark 1 between using a value of KPAR that does not evenly divide the number of kpoints and using a value of KPAR that does evenly divide the number of kpoints. In the KPAR=8 and KPAR=9 graphs below, notice that the 4x4x2 kpoints grid (16 total kpoints) runs much faster at all core counts using KPAR=8 than using KPAR=9. For the 10x10x5 kpoints grid (500 kpoints), 500 is not divisible by either 8 or 9, and we notice that the performance is approximately the same for KPAR=8 and KPAR=9. 
  * We found that runtime scales poorly if you increase the number of nodes past the value of KPAR , so it is recommended to set KPAR no lower than the number of nodes used.
  * Lower values of KPAR might be better for lower node counts. For example, Benchmark 1 calculations on 1-4 nodes were fastest using KPAR=4, but Benchmark 1 calculations on more than 4 nodes were fastest using KPAR=8.
  
Comparing KPAR performance  |  Note that 1 node = 64 cores
:-------------------------:|:-------------------------:
![](https://github.com/claralarson/HPC/blob/45fcff2eb3c13ee8ba318eb7bf7e936229f42db3/applications/vasp/VASP%20Recommendations/Images/Swift_1_K1_N4.png) |  ![](https://github.com/claralarson/HPC/blob/45fcff2eb3c13ee8ba318eb7bf7e936229f42db3/applications/vasp/VASP%20Recommendations/Images/Swift_1_K4_N4.png)
![](https://github.com/claralarson/HPC/blob/45fcff2eb3c13ee8ba318eb7bf7e936229f42db3/applications/vasp/VASP%20Recommendations/Images/Swift_1_K8_N4.png) |  ![](https://github.com/claralarson/HPC/blob/45fcff2eb3c13ee8ba318eb7bf7e936229f42db3/applications/vasp/VASP%20Recommendations/Images/Swift_1_K9_N4.png)
  
  
      
### K-Points Scaling 

Runtime does not scale well with the number of kpoints. Benchmark 1 uses a 10x10x5 kpoints grid (500 kpoints). When run with a 4x4x2 kpoints grid (16 kpoints), we should expect the runtime to scale by 16/500 (3.2%) since calculations are being performed at 16 points rather than 500. However, the average scaling factor between Benchmark 1 jobs on Swift with 10x10x5 grids and 4x4x2 grids is 28% (ranging from ~19%-39%).

### Scripts for Running VASP on Swift
  * [VASP on Swift with Intel MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Swift_IntelMPI.slurm)
  * [VASP on Swift with Open MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Swift_OpenMPI.slurm)
  * [VASP on Swift with Shared Nodes using Intel MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Swift_IntelMPI_shared_nodes.slurm)
  * [VASP on Swift with Shared Nodes using Open MPI](https://github.com/claralarson/HPC/blob/7e2759711664ecdbf78377e07671ed1708791c5e/applications/vasp/VASP%20Recommendations/VASP%20scripts/Swift_OpenMPI_shared_nodes.slurm)
