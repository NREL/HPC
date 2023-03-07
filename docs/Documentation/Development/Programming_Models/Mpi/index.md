# OpenMpi
**Documentation:** [ link to documentation](https://www.open-mpi.org)

*The Open MPI Project is an open source Message Passing Interface implementation that is developed and maintained by a consortium of academic, research, and industry partners. Open MPI is therefore able to combine the expertise, technologies, and resources from all across the High Performance Computing community in order to build the best MPI library available. Open MPI offers advantages for system and software vendors, application developers and computer science researchers.*

## Getting Started

Two version of OpenMpi are available: `openmpi/$version-gcc` and `openmpi/$version-intel`.
The first is compiled with `gcc` and the second with `intel-compilers`.
OpenMpi is intalled on all systems (Eagle, Vermilion, Swift) and can be loaded using 

```
module avail program
   openmpi/4.1.5-gcc     openmpi/4.1.5-intel 
```

```
module load openmpi/4.1.5-gcc
```

## Supported Versions

| Eagle                                | Swift          | Vermillion |
|:------------------------------------:|:--------------:|:----------------:|
openmpi/1.10.7/gcc-8.4.0               |  openmpi/4.1.1 |openmpi/4.1.4-gcc |   
openmpi/3.1.6/gcc-8.4.0                |                |                  |
openmpi/4.0.4/gcc-8.4.0                |                |                  |
openmpi/4.1.1/gcc+cuda                 |||
openmpi/4.1.2/gcc                      |||
openmpi/4.1.2/intel                    |||
openmpi/4.1.3/gcc-11.3.0-cuda-11.7     |||
openmpi/4.1.0/gcc-8.4.0                |||

    

## Advanced

Include advanced user information about the code here (see BerkeleyGW page for some examples)

One common "advanced case" might be that users want to build their own version of the code.

### Building From Source

Please refere to ()[].

## Troubleshooting

Include known problems and workarounds here, if applicable



# Mpich
**Documentation:** [ link to documentation](https://www.mpich.org)

*MPICH is a high performance and widely portable implementation of the Message Passing Interface (MPI) standard. 
MPICH and its derivatives form the most widely used implementations of MPI in the world. They are used exclusively on nine of the top 10 supercomputers (June 2016 ranking), including the world’s fastest supercomputer: Taihu Light.*

## Getting Started

Two version of Mpich are available: `mpich/$version-gcc` and `mpich/$version-intel`.
The first is compiled with `gcc` and the second with `intel-compilers`.
OpenMpi is intalled on all systems (Eagle, Vermilion, Swift) and can be loaded using 

```
module avail program
   openmpi/4.1.5-gcc     openmpi/4.1.5-intel 
```

```
module load openmpi/4.1.5-gcc
```

## Supported Versions

| Eagle                                | Swift          | Vermillion |
|:------------------------------------:|:--------------:|:----------------:|
openmpi/1.10.7/gcc-8.4.0               |  openmpi/4.1.1 |openmpi/4.1.4-gcc |   
openmpi/3.1.6/gcc-8.4.0                |                |                  |
openmpi/4.0.4/gcc-8.4.0                |                |                  |
openmpi/4.1.1/gcc+cuda                 |||
openmpi/4.1.2/gcc                      |||
openmpi/4.1.2/intel                    |||
openmpi/4.1.3/gcc-11.3.0-cuda-11.7     |||
openmpi/4.1.0/gcc-8.4.0                |||

    

## Advanced

Include advanced user information about the code here (see BerkeleyGW page for some examples)

One common "advanced case" might be that users want to build their own version of the code.

### Building From Source

Please refere to ()[].

## Troubleshooting

Include known problems and workarounds here, if applicable



