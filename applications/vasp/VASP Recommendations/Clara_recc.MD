# Table of Contents
1. [VASP Performance on Eagle vs Swift](#VASP-Performance-on-Eagle-vs-Swift)
2. [Eagle](#Eagle)
    * [Recommendations for Running VASP on Eagle](#Recommendations-for-running-VASP-on-Eagle)
    * [Scripts for VASP on Eagle](#Scripts-for-VASP-on-Eagle)
3. [Swift](#Swift)
    * [Recommendations for Running VASP on Swift](#Recommendations-for-running-VASP-on-Swift)
    * [Scripts for VASP on Swift](#Scripts-for-VASP-on-Swift)

## VASP Performance on Eagle vs Swift

Eagle vs. Swift full nodes plot

## Eagle

### Recommendations for Running VASP on Eagle

Run VASP on full nodes (36 CPUs/node). While using fewer cores per node yields an improvement in runtime/core, it will results in a larger allocation charge, as Eagle charges per node used, regardless of how much of the node is used. Eagle charges 3AUs/node-hour. 

### VASP on GPU Nodes

Running the GPU build of VASP (vasp_gpu) on GPU nodes improves performance for larger VASP calculations, but may increase the runtime for smaller calculations. Benchmark 2 calculations on GPU nodes ran in an average of 27.5% of the time as CPU calculations on the same number of nodes, but Benchmark 1 calculationson GPU nodes ran in an average of 150% of the time as CPU calculations on the same number of nodes.
   * Memory limitation: GPU nodes on Eagle cannot provide as much memory as CPU nodes for VASP jobs, and large VASP jobs may require more GPU nodes to provide enough memory for the calculation. For Benchmark 2, at least 2 full nodes were needed to provide enough memory to complete a calculation. Using more complicated parallelization schemes, the number of nodes necessary to provide enough memory scaled with the increase in number of problems handled simultaneousely. 

### MPI

Intel MPI is recommended over Open MPI. Using an Intel MPI build of VASP and running over Intel MPI, Benchmark 2 ran in average of 50% of the time as the same calculations using an Open MPI build of VASP over Open MPI. For Benchmark 1, Intel MPI calculations ran in an average of 60% of the time as Open MPI calculcations. 

### --cpu-bind Flag
Setting --cpu-bind=cores or rank showed no improvement in the performance of VASP on 36 CPUs/node. Using 18 CPUs/node setting --cpu-bind=cores shows a small improvement (~5% CHECK WITH OPENMPI RESULTS TOO) in runtime. (See [LINK TO CPU-BIND DOC] for info on the effect of cpu-bind)

cpu-bind can be set as a flag in an srun command, such as 
```
srun --cpu-bind=cores vasp_std
```
### KPAR

KPAR determines the number of groups across which to divide calculations at each kpoint, and one calculation from each group is performed at a time. The value of KPAR can be defined in the INCAR file. 
      - Per [VASP documentation](https://www.vasp.at/wiki/index.php/KPAR), the KPAR tag file should always be set to a value that evenly divides the total number of cores used. 
      - We found that runtime starts to increase (in other words, runtime scales positively with the number of nodes, rather than negatively) if you increase the number of nodes past the value of KPAR, so it is recommended to set KPAR no lower than the number of nodes used. 
      - Lower values of KPAR might be better for lower node counts. For example, Benchmark 1 calculations on 1-4 nodes were fastest using KPAR=4, but Benchmark 1 calculations on more than 4 nodes were fastest using KPAR=9.
      - The [GPU VASP documentation](https://www.vasp.at/wiki/index.php/OpenACC_GPU_port_of_VASP) recommends setting KPAR equal to the total number of GPUs used. Our results are relatively consistent with this recommendation. 

### K-Points Scaling

Runtime does not scale well with the number of kpoints. Benchmark 1 uses a 10x10x5 kpoints grid (500 kpoints). When run with a 4x4x2 kpoints grid (16 kpoints), we should expect the runtime to scale by 16/500 (3.2%) since calculations are being performed at 16 points rather than 500. However, the average scaling factor between Benchmark 1 jobs on Eagle with 10x10x5 grids and 4x4x2 grids is 26%. 

Scripts to add:
- IntelMPI VASP on Eagle script
- OpenMPI VASP on Eagle script
- VASP on Eagle GPU nodes script (Intel)

Plots to include:
- GPUs plot?

## Swift

### Recommendations for Running VASP on Swift

On Swift, VASP is most efficiently run on partially full nodes. 32 CPUs/node was found to have the fastest runtime/core, followed by 64 CPUs/node and 128 CPUs/node. Compared to jobs on 64 CPUs/node, jobs on 32 CPUs/node using the same total number of cores ran in 70%-90% of the 64 CPUs/node runtime. Unlike on Eagle, Swift charges for only the portion of the node requested by a job. Swift charges 5 AU/hour when running on 128 nodes (one full node), so running on 32 CPUs, for example, would charge only (32/128) * 5 AUs/hour rather than the full 5 AUs/node-hour. 

### MPI

Intel MPI is recommended over Open MPI for all VASP calculations on Swift. Using an Intel MPI build of VASP and running over Intel MPI, Benchmark 2 ran in average of 76%, 72% and 46% of the time as the same calculations using an Open MPI build of VASP over Open MPI on 32, 64 and 128 CPUs/node, respectively. For Benchmark 1, Intel MPI calculations ran in an average of CHECK BACK of the time as Open MPI calculcations. 

### --cpu-bind Flag

On Swift, it is recommended not to use cpu-bind. Running VASP on 64 CPUs/node and 128 CPUs/node, setting --cpu-bind=cores or rank showed no improvement in runtime. Running VASP on 32 CPUs/node, setting --cpu-bind=cores or rank increased runtime by up to 40%. (See [LINK TO CPU-BIND DOC] for info on the effect of cpu-bind)

```
srun --cpu-bind=cores vasp_std
```

### KPAR

KPAR determines the number of groups across which to divide calculations at each kpoint, and one calculation from each group is performed at a time. The value of KPAR can be defined in the INCAR file. 
      - Per [VASP documentation](https://www.vasp.at/wiki/index.php/KPAR), the KPAR tag file should always be set to a value that evenly divides the total number of cores used. 
      - We found that runtime starts to increase (in other words, runtime scales positively with the number of nodes, rather than negatively) if you increase the number of nodes past the value of KPAR, so it is recommended to set KPAR no lower than the number of nodes used. 
      - Lower values of KPAR might be better for lower node counts. For example, Benchmark 1 calculations on 1-4 nodes were fastest using KPAR=4, but Benchmark 1 calculations on more than 4 nodes were fastest using KPAR=9.
      
### K-Points Scaling 

Runtime does not scale well with the number of kpoints. Benchmark 1 uses a 10x10x5 kpoints grid (500 kpoints). When run with a 4x4x2 kpoints grid (16 kpoints), we should expect the runtime to scale by 16/500 (3.2%) since calculations are being performed at 16 points rather than 500. However, the average scaling factor between Benchmark 1 jobs on Swift with 10x10x5 grids and 4x4x2 grids is 23%. 

Scripts to add:
- IntelMPI VASP on Swift script
- OpenMPI VASP on Swift script
- Running on split nodes as an array job

Plots to include:
- nodefill plot
- some KPAR/NPAR plots that show performance drops off after node count meets KPAR
