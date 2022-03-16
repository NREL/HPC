# VASP Recommendations

In this study, the ESIF VASP Benchmarks 1 and 2 were used. Benchmark 1 is a system of 16 atoms, and Benchmark 2 is a system of 519 atoms. Benchmark 2 was used to explore differences in runtimes between running on half-filled and full nodes on Swift anf Eagle as well as runtime improvements due to running on Eagle's GPU nodes. Since Benchmark 1 represents a smaller system and requires less computational time to run, all Benchmark 1 calculations were run using 4x4x2 and 10x10x5 kpoints grids in order to measure kpoints scaling and parallelization on Swift and Eagle. Additionally, Benchmark 1 was run with 8 different KPAR and NPAR configurations in the INCAR file in order to explore the efficiency of various VASP parallization schemes on Swift and Eagle. Both Benchmarks were used to compare runtimes between IntelMPI and OpenMPI and across various cpu-bind settings.

Average scaling rates between two parameter settings were calculated by first calculating the average runtime at every core count for each setting, calculating the percent difference between the two settings at each core count, and then averaging the set of percentages across all core counts. Since differences in runtime tend to be larger at higher core counts, this was done to give each core count equal weight in the overall average.

* [Recommendations for Running VASP on Eagle](#Recommendations-for-Running-VASP-on-Eagle)

1. [Recommendations for Running VASP on Eagle](#Recommendations-for-Running-VASP-on-Eagle)
2. [Recommendations for Running VASP on Swift](#Recommendations-for-Running-VASP-on-Swift)

## Recommendations for Running VASP on Eagle

Cores per Node:
> Running on half-full nodes yields better runtime per core used. Using Benchmark 2 on Eagle, running on half-full nodes used an average of 81% of the runtime used to run on full nodes with the same total number of cores.

Using GPUs:
> Running on two GPUs per node significantly inproves runtime performance using Benchmark 2. Reference graph to see the extent of improvement. 

MPI:
> Based on average runtimes over all node counts, there is little difference between running with Intel MPI or OpenMPI. However, for both Benchmark 1 and Benchmark 2, the graphs show that using OpenMPI may improve runtimes for multi-node jobs on full nodes. OpenMPI does not improve runtimes for multi-node jobs on half-full nodes.

cpu-bind:
> Best cpu-bind performance by calculation type:
>> - Full nodes: Based on average runtimes over all node counts, --cpu-bind doesn't have much of an effect on results. However, the graphs show that setting --cpu-bind=rank may improve runtimes for jobs on higher node counts (4+ nodes) using both Benchmark 1 and Benchmark 2.
>> - Half-filled nodes: --cpu-bind=cores runs in an average of 92% percent of the time as calculations with no cpu-bind set using Benchmark 1. The runtime improvement is consistent across all node counts. 
>> - GPU nodes: Setting --cpu-bind=[either rank or cores] yields worse runtimes than having no --cpu=bind setting, using Benchmark 1.

KPOINTS Scaling:
> Benchmark 1 was run using both a 4x4x2 kpoints grid (32 kpoints) and a 10x10x5 kpoints grid (500 kpoints). All Benchmark 1 calculations were run using full nodes. We should expect the runtime to scale proportionally to the change in kpoints, so we would expect the 4x4x2 kpoints grid calculations to run in 6.4% (32/500) of the amount of time needed to run the the 10x10x5 kpoints grid calculations. However, we found that the 4x4x2 kpoints grid calculations ran, on average, in 26.19% of the amount of time needed to run the 10x10x5 kpoints grid calculations. In fact, using the best performing values of KPAR and NPAR, the 4x4x2 kpoints grid calculations ran in 41% of the amount of time needed to run the 10x10x5 kpoints grid calculations. Overall, we found that using a smaller kpoints grid does not yield the expected decreases in runtime.

> For each combination of KPAR and NPAR, the table below gives the average amount of the time needed to run each 4x4x2 calulcation expressed as a percentage of the time needed to run the corresponding 10x10x5 calculations. The default KPAR/NPAR configuration (KPAR=1, NPAR=# of cores) yields the best KPOINTS scaling, but the slowest overall runtimes. 

KPAR and NPAR:
> All KPAR/NPAR results are from Benchmark 1 calculations on full nodes. For each combination of KPAR and NPAR used, the "Comparison to Default" columnns in the table below gives the average amount of the time needed to run using the given KPAR/NPAR configuration expressed as a percentage of the time needed to run the corresponding calculations with the default KPAR/NPAR settings (KPAR=1, NPAR=# of cores). Seperate averages were done for calculations with 4x4x2 kpoint grids and those with 10x10x5 kpoints grids. 
> Based on average runtimes across all core counts, KPAR=4, NPAR=sqrt(#of cores) is the best performing configuration of KPAR and NPAR, followed by the other two configurations with KPAR=4. However, the extent to which the runtime improves as core count increase is lost in taking the average - KPAR=9, NPAR=4 is the configuration that reaches the fastest runtime at high core counts. 

> Configurations that don't perform as well at higher node/core counts:
>> - KPAR = 1, NPAP = 4
>> - KPAR = 1, NPAR = # of cores
>> - KPAR = 1, NPAR = sqrt(# of cores)

> Configurations that do perform pretty well on higher node/core counts:
>> - KPAR = 4, NPAR = # of cores
>> - KPAR = 9, NPAR =4
>> - KPAR = 4, NPAR = sqrt(# of cores)
>> - KPAR = 4, NPAR = 4
>> - KPAR = 9, NPAR = 9


|     | Average 4x4x2 Runtime as a Percentage of 10x10x5 Runtime | Average Runtime as a Percentage of default KPAR/NPAR Configuration Runtime (4x4x2) | Average Runtime as a Percentage of default KPAR/NPAR Configuration Runtime (10x10x5) |
| ----------- | ----------- | ----------- | ----------- |
| KPAR=1,NPAR=4    | 21.59%       | 64.31% | 71.27% |
| KPAR=1,NPAR=# of cores    | 20.15%       | Default | Default |
| KPAR=1,NPAR=sqrt(# of cores)    | 26.15%       |  78.48% | 69.92% |
| KPAR=4,NPAR=# of cores    | 26.28%       | 36.61% | 40.97% |
| KPAR=9,NPAR=4    | 41.41%       | 70.91% | 58.44% |
| KPAR=4,NPAR=sqrt(# of cores)    | 22.62%       | 35.38% | 56.50% |
| KPAR=4,NPAR=4    | 23.83%       | 36.29% | 42.71% |
| KPAR=9,NPAR=sqrt(# of cores)    | 27.51%       | 71.88% | 59.42% |
| Average    | 26.19%       |   |  |

## Recommendations for Running VASP on Swift

Cores per Node:
> Cores per Node performance for Intel MPI and OpenMPI seperately since they perform drastically different on Swift:
>> - Using Intel MPI: Running on half-full nodes yields better runtime per core used. Using Benchmark 2 on Swift with Intel MPI, running on half-full nodes used an average of 83.73% of the runtime used to run on full nodes with the same total number of cores.
>> - Using OpenMPI: Running on half-full nodes yields better runtime per core used. Using Benchmark 2 on Swift with OpenMPI, running on half-full nodes used an average of 74.60% of the runtime used to run on full nodes with the same total number of cores.

MPI:
> Best MPI performance by calculation type:
>> - Full nodes: Calculations on full nodes run with Intel MPI have significantly faster runtimes than calculations run with OpenMPI using both Benchmark 1 and Benchmark 2 on Swift. For Benchmark 2, Intel MPI calculations run in an average of 52.41% of the time needed for OpenMPI calculations. For Benchmark 1, Intel MPI calculations run in an average of 84.49% of the time needed for OpenMPI calculations using 1 4x4x2 kpoints grid, and Intel MPI calculations run in an average of 85.82% of the time needed for OpenMPI calculations using 1 10x10x5 kpoints grid.
>> - Half-filled nodes: Calculations on half-filled nodes run with Intel MPI have significantly faster runtimes than calculations run with OpenMPI using both Benchmark 1 and Benchmark 2 on Swift. For Benchmark 2, Intel MPI calculations run in an average of 71.77% of the time needed for OpenMPI calculations.

cpu-bind:
> Best cpu-bind performance by calculation type:
>> - Full nodes: On average, calculations with --cpu-bind=rank run in 88.53% of the time as calculations with no --cpu-bind using Benchmark 2, but --cpu-bind did not affect runtimes using Benchmark 1. 
>> - Half-filled nodes: Setting --cpu-bind=[either rank or cores] yields much slower runtimes than setting no --cpu-bind using Benchmark 2. 

KPOINTS Scaling:
> Benchmark 1 was run using both a 4x4x2 kpoints grid (32 kpoints) and a 10x10x5 kpoints grid (500 kpoints). All Benchmark 1 calculations were run using full nodes. We should expect the runtime to scale proportionally to the change in kpoints, so we would expect the 4x4x2 kpoints grid calculations to run in 6.4% (32/500) of the amount of time needed to run the the 10x10x5 kpoints grid calculations. However, we found that the 4x4x2 kpoints grid calculations ran, on average, in 23.03% of the amount of time needed to run the 10x10x5 kpoints grid calculations. In fact, using the best performing values of KPAR and NPAR, the 4x4x2 kpoints grid calculations ran in 34.10% of the amount of time needed to run the 10x10x5 kpoints grid calculations. Overall, we found that using a smaller kpoints grid does not yield the expected decreases in runtime.

> For each combination of KPAR and NPAR, the table below gives the average amount of the time needed to run each 4x4x2 calulcation expressed as a percentage of the time needed to run the corresponding 10x10x5 calculations. The default KPAR/NPAR configuration (KPAR=1, NPAR=# of cores) yields one of the best KPOINTS scaling, but the slowest overall runtimes. 

KPAR and NPAR:
> All KPAR/NPAR results are from Benchmark 1 calculations on full nodes. For each combination of KPAR and NPAR used, the table below gives the average amount of the time needed to run using the given KPAR/NPAR configuration expressed as a percentage of the time needed to run the corresponding calculations with the default KPAR/NPAR settings (KPAR=1, NPAR=# of cores). Seperate averages were done for calculations with 4x4x2 kpoint grids and those with 10x10x5 kpoints grids. 

> Based on average runtimes across all core counts, KPAR=9, NPAR=4 and  KPAR=4, NPAR=4 have the best runtimes. However, the KPAR=9, NPAR=4 is the only one with fast runtimes at high core counts. All other configurations see runtimes increase as core counts increase, which is not expected. KPAR=9, NPAR=4 performs worse than all three of the KPAR=4 configurations on low core counts (1 or 2 nodes). For calculations on lower node counts, the KPAR=4, NPAR=4 configuration has the fastest runtimes. 

> Configurations that perform best at lower node/core counts (1 or 2 nodes):
>> - KPAR = 4, NPAR = # of cores
>> - KPAR = 4, NPAR = sqrt(# of cores)
>> - KPAR = 4, NPAR = 4

> Configurations that perform best on higher node/core counts (3+ nodes):

>> - KPAR = 9, NPAR =4

|     | Average 4x4x2 Runtime as a Percentage of 10x10x5 Runtime | Average Runtime as a Percentage of default KPAR/NPAR Configuration Runtime (4x4x2) | Average Runtime as a Percentage of default KPAR/NPAR Configuration Runtime (10x10x5) |
| ----------- | ----------- | ----------- | ----------- |
| KPAR=1,NPAR=4    | 19.75%       | 50.90% | 60.28% |
| KPAR=1,NPAR=# of cores    | 18.76%       | Default | Default |
| KPAR=1,NPAR=sqrt(# of cores)    | 18.56%       |  89.92% | 89.36% |
| KPAR=4,NPAR=# of cores    | 20.46%       | 31.55% | 35.68% |
| KPAR=9,NPAR=4    | 34.10%       | 37.86% | 24.23% |
| KPAR=4,NPAR=sqrt(# of cores)    | 19.94%       | 37.25% | 43.02% |
| KPAR=4,NPAR=4    | 20.70%       | 29.76% | 33.77% |
| KPAR=9,NPAR=sqrt(# of cores)    | 31.93%       | 34.96% | 50.81% |
| Average    | 23.03%       |   |  |
