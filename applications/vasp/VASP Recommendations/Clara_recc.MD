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
