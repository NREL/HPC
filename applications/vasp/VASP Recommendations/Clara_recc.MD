# Table of Contents
1. [VASP Performance on Eagle vs Swift](#VASP-Performance-on-Eagle-vs-Swift)
2. [Eagle](#Eagle)
    * [Recommendations for Running VASP on Eagle](#Recommendations-for-running-VASP-on-Eagle)
    * [Summary of Eagle Data Analysis](#Summary-of-Eagle-Data-Analysis)
3. [Swift](#Swift)
    * [Recommendations for Running VASP on Swift](#Recommendations-for-running-VASP-on-Swift)
    * [Summary of Swift Data Analysis](#Summary-of-Swift-Data-Analysis)

## VASP Performance on Eagle vs Swift

Eagle vs. Swift full nodes plot

## Eagle

### Recommendations for Running VASP on Eagle

   * Run VASP on full nodes (36 CPUs/node). While using fewer cores per node yields an improvement in runtime/core, it will results in a larger allocation charge, as Eagle charges per node used, regardless of how much of the node is used.
   * <strong>GPU Nodes:</strong> Running the GPU build of VASP (vasp_gpu) on GPU nodes improves performance for larger VASP calculations, but may increase the runtime for smaller calculations. Benchmark 2 calculations on GPU nodes ran in an average of 27.5% of the time as CPU calculations on the same number of nodes, but Benchmark 1 calculationson GPU nodes ran in an average of 150% of the time as CPU calculations on the same number of nodes. 
      - GPU nodes on Eagle cannot provide as much memory as CPU nodes for VASP jobs, and large VASP jobs may require more GPU nodes to provide enough memory for the calculation. For Benchmark 2, at least 2 full nodes were needed to provide enough memory to complete a calculation. Using more complicated parallelization schemes, the number of nodes necessary to provide enough memory scaled with the increase in number of problems handled simultaneousely. 
   * <strong>MPI:</strong> Intel MPI is recommended over Open MPI. Using an Intel MPI build of VASP and running over Intel MPI, Benchmark 2 ran in average of 50% of the time as the same calculations using an Open MPI build of VASP over Open MPI. For Benchmark 1, Intel MPI calculations ran in an average of 60% of the time as Open MPI calculcations. 
   * <strong>cpu-bind:</strong> Setting --cpu-bind = rank may improve the performance of VASP on full nodes by up to 10%. Using half-full nodes, setting --cpu-bind=cores shows a comprable improvement. 
   * <strong>KPAR:</strong> KPAR determines the number of groups across which to divide calculations at each kpoint, and one calculation from each group is performed at a time.
      - Per [VASP documentation](https://www.vasp.at/wiki/index.php/KPAR), the KPAR tag in the INCAR file should always be set to a value that evenly divides the total number of cores used. 
      - We found that runtime starts to increase (in other words, runtime scales positively with the number of nodes, rather than negatively) if you increase the number of nodes past the value of KPAR, so it is recommended to set KPAR no lower than the number of nodes used. 
      - Lower values of KPAR might be better for lower node counts. For example, Benchmark 1 calculations on 1-4 nodes were fastest using KPAR=4, but Benchmark 1 calculations on more than 4 nodes were fastest using KPAR=9.
      - The [GPU VASP documentation](https://www.vasp.at/wiki/index.php/OpenACC_GPU_port_of_VASP) recommends setting KPAR equal to the total number of GPUs used. Our results are relatively consistent with this recommendation. 

### Summary of Eagle Data Analysis

Cores scaling within a node

Provide stats for:
- Full nodes (1,2)
- Half-filled nodes (2)
- GPU nodes (1,2) 
  * discuss the difference between the two GPU builds
  * discuss the memory limitations of the OpenACC build
- Table with runtime for half-filled, full, and GPU nodes and discuss allocation and accessibility?

--> current data: averages GPU build for KPAR=1, averages full nodes over all KPAR/NPAR combos

### Average Total Runtime (s) to Complete One Job on Eagle Using Benchmark 2
|     | IntelMPI, 1 node | IntelMPI, 2 nodes | IntelMPI, 4 nodes | OpenMPI, 1 node |OpenMPI, 2 nodes |OpenMPI, 4 nodes| AUs Charged? |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |----------- |----------- |
| Full Nodes      | 6300.00            |             3174    |       2341.33    |8287.67           | 5338.00     |   4277.33    |
| Half-filled Nodes     | 10627.67           |             5083.67 |             2665.33 |                  |                   |       |
| GPU Nodes     |                    |              922    |              633     |                   |               |       |
| Half/Full      |   1.69               |                1.6  |           1.14        |                  |              |       |
| GPU/Full      |                   |                0.29 |                 0.27  |               |                   |

### Average Total Runtime (s) to Complete One Job on Eagle Using Benchmark 1
|     | IntelMPI, 1 node | IntelMPI, 2 nodes | IntelMPI, 4 nodes | OpenMPI, 1 node |OpenMPI, 2 nodes |OpenMPI, 4 nodes| AUs Charged? |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |----------- |----------- |
| Full Nodes      |   501.08 |          340.25 |          359.5  | 612.50         | 357.67         | 231.00   |            |
| GPU Nodes     |           804.33 |          523.33 |          521  |                |                |      |       |
| GPU/Full      |             1.61 |            1.54 |            1.45|                |                |             |                   |


Give best recommendations for (Discuss how it changes for different node fills?): 
- MPI (1,2)
- cpu-bind (1,2)
- KPAR/NPAR (1)

Discuss:
- KPOINTS scaling (1)

Scripts to add:
- IntelMPI VASP on Eagle script
- OpenMPI VASP on Eagle script
- VASP on Eagle GPU nodes script

Plots to include:
- nodefill plot
- MPI plot

## Swift

### Recommendations for Running VASP on Swift

### Summary of Swift Data Analysis

Cores scaling within a node

Provide stats for:
- Full nodes (1,2)
- half-filled nodes (2)
- split nodes (2)
- virtual nodes
- Table with runtime for half-filled, full, split nodes and virtual nodes and discuss allocation and accessibility?

### Average Total Runtime (s) to Complete Two Jobs on Swift Using Benchmark 2
|     | IntelMPI, 1 node | IntelMPI, 2 nodes |IntelMPI, 4 nodes | OpenMPI, 1 node | OpenMPI, 2 nodes |OpenMPI, 4 nodes | AUs Charged? |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |----------- | ----------- |
|  Shared Nodes  |  23771.7  |        11183    |                 |       15392.2  |       14295.5  |    |        |
| Full Nodes      |  13226    |         8527    | 5818.67         |       17348    |       14962.7  | 11962.67        |       |
| Both Virtual Nodes      | 13314    |         9612    | 6998.00         |       22315.3  |       19731.3  | 19918.00  |      |
| Half-filled Nodes     |  18866    |        11918    | 6504.00         |       19835.6  |       13857.3  | 11006.00 |       |
| Shared/Full      |     1.8  |            1.31 |                 |           0.89 |           0.96 |      |       |
| Virtual/Full     |      1.01 |            1.13 | 1.20            |           1.29 |           1.32 | 1.67           |       |
| Half/Full      |    1.43 |            1.4  | 1.12            |           1.14 |           0.93 | 0.92|       |

Give best recommendations for (Discuss how it changes for different node fills?):
- MPI (1,2)
- cpu-bind (1,2)
- KPAR/NPAR

Discuss:
- KPOINTS scaling (1)

Scripts to add:
- IntelMPI VASP on Swift script
- OpenMPI VASP on Swift script
- Running on split nodes as an array job
- half-filled and virtual? (Really this is just changing -ntasks, but maybe make a note of how to use the --exclusive tag)

Plots to include:
- nodefill plot
- MPI plot
