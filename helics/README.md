*<sub>by Slava Barsuk</sub>*

# Hierarchical Engine for Large-scale Infrastructure Co-Simulation (HELICS)


This procedure describes how to build softare with modules. This tutorial will cover compiling HELICS v2.0.0 with MPI support for NREL HPC Systems.

## Environment Setup

HELICS is built with cmake, so it is very important to have a clean environment.
Do not mix module and conda enviroment,as it may change the search order of library discovery for `cmake`

Required modules for helics build:
```
 boost/1.69.0/gcc-7.3.0
 gcc/7.3.0
 openmpi/3.1.3/gcc-7.3.0
 cmake/3.12.3 
```
For this particular build helics module was created:

Now create a [`helics-2.0.0.lua`](helics-2.0.0.lua) file and populate it with these contents:

<embed src="helics-2.0.0.lua">

```lua
help([[****helics cosimulation software]])

whatis("Name: helics")
whatis("Version: 2.0.0")

local datestamp = "2019-04-30"
local base = "/projects/aces/helics_2.0.0_mpi"

load("boost/1.69.0/gcc-7.3.0")
load("gcc/7.3.0")
load("openmpi/3.1.3/gcc-7.3.0")

prepend_path("CPATH", pathJoin(base, "include"))
prepend_path("CMAKE_PREFIX_PATH", base)
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib64"))
prepend_path("LIBRARY_PATH", pathJoin(base, "lib64"))
prepend_path("LD_LIBRARY_PATH",pathJoin(base,"lib"))
prepend_path("LIBRARY_PATH",pathJoin(base,"lib"))
prepend_path("PATH",pathJoin(base,"bin"))
prepend_path("PYTHONPATH",pathJoin(base,"python"))
```

Correct CMAKE_PREFIX_PATH is important for the proper libraries discovery by cmake

For some reason helics build is not friendly with Intel icc compiler. I was not able to make a clean build with icc

1. Create directory structure 

helics installation was build for the following directory:

/projects/aces/helics_2.0.0_mpi


3. Install ZeroMQ

Helics requires ZeroMQ by default. It is easier to incorporate ZeroMQ into helics build as a part of it.
Create subdirectory for ZeroMQ and download the ZeroMQ distribution package there.
Here is an example for 4.3.1 build:

mkdir zmq
cp zeromq-4.3.1.tar.gz zmq
cd zmq
tar -xzvf zeromq-4.3.1.tar.gz
cd zeromq-4.3.1

   there check for INSTALL and readme file for build procedure changes

./configure --prefix=/projects/aces/helics_2.0.0_mpi

--prefix is important and points to ZeroMQ installation directory

make
make install

it should create  a few directories under /projects/aces/helics_2.0.0_mpi
we need
  ./lib
  ./include

You would have to add the discovery path for cmake for ZeroMQ in those directories
in modules it can be implemented like:


4. Download helics-2.0.0 distribution

cd /projects/aces/helics_2.0.0_mpi
git clone https://github.com/GMLC-TDC/HELICS-src

  it will create HELICS-src subdirectory

cd HELICS-src

5. Update/Verify CMakeLists.txt file

CMakeLists.txt file by default was not supporting proper c++ discovery at the time of build
make sure that fragment of CMakeLists.txt file looks like this:


# -------------------------------------------------------------
# finding MPI
# -------------------------------------------------------------

option(ENABLE_MPI "Enable MPI networking library" OFF)
if(ENABLE_MPI)
    include(addMPI)
    if(MPI_CXX_FOUND)
        set(HELICS_HAVE_MPI TRUE)
        target_link_libraries(helics_base INTERFACE MPI::MPI_CXX)
    endif(MPI_CXX_FOUND)
    if(MPI_C_FOUND)
        set(HELICS_HAVE_MPI TRUE)
        target_link_libraries(helics_base INTERFACE MPI::MPI_C)
    else()
        message(SEND_ERROR "MPI not found")
    endif(MPI_C_FOUND)
else(ENABLE_MPI)
    set(HELICS_HAVE_MPI FALSE)
endif(ENABLE_MPI)



Check that MPI_CXX is there
 
6. generate make files with cmake

mkdir build
cd build
cmake -DENABLE_MPI=ON -DCMAKE_INSTALL_PREFIX=/projects/aces/helics_2.0.0_mpi ..

Check the output for errors, make sure that cmake found Boost, MPI_C, MPI_CXX, ZeroMQ library and ZeroMQ headers:

shoud look like 

-- Boost version: 1.69.0
-- Found the following Boost libraries:
--   program_options
--   filesystem
--   system
--   unit_test_framework
-- Found MPI_C: /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-3.1.3-vfrud3sm7xubb2rtfvkkb4or5p6jgmil/lib/libmpi.so (found version "3.1") 
-- Found MPI_CXX: /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-3.1.3-vfrud3sm7xubb2rtfvkkb4or5p6jgmil/lib/libmpi_cxx.so (found version "3.1") 
-- Found MPI: TRUE (found version "3.1")  
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Check if compiler accepts -pthread
-- Check if compiler accepts -pthread - yes
-- Found Threads: TRUE  
-- Found ZeroMQ library: /projects/aces/helics_2.0.0_mpi/lib/libzmq.so
-- Found ZeroMQ headers: /projects/aces/helics_2.0.0_mpi/include


7. Actual build

make

 -- better be no warnings and no errors

make install

 -- will copy files of successful build into /projects/aces/helics_2.0.0_mpi 
 and will create lib64 subdirectory

check the buid by running helics_broker

cd /projects/aces/helics_2.0.0_mpi/bin
./helics_broker --version

check that helics_broker points to libraries you want:
ldd helics_broker

linux-vdso.so.1 =>  (0x00007ffe5ac32000)
	libboost_program_options.so.1.69.0 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/boost-1.69.0-yg4dkgp3amzwqyo4bwl5sew6x2k66x6r/lib/libboost_program_options.so.1.69.0 (0x00007fe5479a8000)
	libboost_filesystem.so.1.69.0 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/boost-1.69.0-yg4dkgp3amzwqyo4bwl5sew6x2k66x6r/lib/libboost_filesystem.so.1.69.0 (0x00007fe54778c000)
	libboost_system.so.1.69.0 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/boost-1.69.0-yg4dkgp3amzwqyo4bwl5sew6x2k66x6r/lib/libboost_system.so.1.69.0 (0x00007fe547588000)
	libmpi_cxx.so.40 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-3.1.3-vfrud3sm7xubb2rtfvkkb4or5p6jgmil/lib/libmpi_cxx.so.40 (0x00007fe54736c000)
	libmpi.so.40 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-3.1.3-vfrud3sm7xubb2rtfvkkb4or5p6jgmil/lib/libmpi.so.40 (0x00007fe546ecd000)
	librt.so.1 => /usr/lib64/librt.so.1 (0x00007fe546cc5000)
	libzmq.so.5 => /projects/aces/helics_2.0.0_mpi/lib/libzmq.so.5 (0x00007fe546a32000)
	libstdc++.so.6 => /nopt/nrel/apps/compilers/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/gcc-7.3.0-vydnujncq3lpwhhnxmauinsqxkhxy4gn/lib64/libstdc++.so.6 (0x00007fe5466af000)
	libm.so.6 => /usr/lib64/libm.so.6 (0x00007fe5463ad000)
	libgcc_s.so.1 => /nopt/nrel/apps/compilers/spack/opt/spack/linux-centos7-x86_64/gcc-4.8.5/gcc-7.3.0-vydnujncq3lpwhhnxmauinsqxkhxy4gn/lib64/libgcc_s.so.1 (0x00007fe546196000)
	libpthread.so.0 => /usr/lib64/libpthread.so.0 (0x00007fe545f79000)
	libc.so.6 => /usr/lib64/libc.so.6 (0x00007fe545bb6000)
	/lib64/ld-linux-x86-64.so.2 (0x000055d8a56fa000)
	liblustreapi.so.1 => /usr/lib64/liblustreapi.so.1 (0x00007fe54598a000)
	libopen-rte.so.40 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-3.1.3-vfrud3sm7xubb2rtfvkkb4or5p6jgmil/lib/libopen-rte.so.40 (0x00007fe545648000)
	libopen-pal.so.40 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/openmpi-3.1.3-vfrud3sm7xubb2rtfvkkb4or5p6jgmil/lib/libopen-pal.so.40 (0x00007fe5451e4000)
	librdmacm.so.1 => /usr/lib64/librdmacm.so.1 (0x00007fe544fc8000)
	libibverbs.so.1 => /usr/lib64/libibverbs.so.1 (0x00007fe544dae000)
	libpmi2.so.0 => /nopt/slurm/18.08.3/lib/libpmi2.so.0 (0x00007fe544b96000)
	libpmi.so.0 => /nopt/slurm/18.08.3/lib/libpmi.so.0 (0x00007fe544990000)
	libslurmfull.so => /nopt/slurm/18.08.3/lib/slurm/libslurmfull.so (0x00007fe5445d0000)
	libutil.so.1 => /usr/lib64/libutil.so.1 (0x00007fe5443cc000)
	libhwloc.so.5 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/hwloc-1.11.11-3gk62vkvspdjzihglti2anxaagc3z6xt/lib/libhwloc.so.5 (0x00007fe54418b000)
	libnuma.so.1 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/numactl-2.0.12-6jal7s2i7qth3rkzxfxzconekj3itbdw/lib/libnuma.so.1 (0x00007fe543f7f000)
	libpciaccess.so.0 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/libpciaccess-0.13.5-h4jagxkevyt3ajuu3anql2w2cpe2jrqm/lib/libpciaccess.so.0 (0x00007fe543d76000)
	libcudart.so.10.0 => /nopt/nrel/apps/cuda/10.0.130/lib64/libcudart.so.10.0 (0x00007fe543afc000)
	libxml2.so.2 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/libxml2-2.9.8-aeesmb67b4t4ikg52hqlxiqeykhysfnx/lib/libxml2.so.2 (0x00007fe543797000)
	libdl.so.2 => /usr/lib64/libdl.so.2 (0x00007fe543593000)
	libz.so.1 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/zlib-1.2.11-7nzryhnemdpefn6ycyhvqu62cb4asd7x/lib/libz.so.1 (0x00007fe54337c000)
	liblzma.so.5 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/xz-5.2.4-criid5veupvjoh7ubqapeab6sjncowxj/lib/liblzma.so.5 (0x00007fe543155000)
	libiconv.so.2 => /nopt/nrel/apps/base/2019-01-02/spack/opt/spack/linux-centos7-x86_64/gcc-7.3.0/libiconv-1.15-ohrthv5d5erspeqa2mrfnw2rhtacdkn7/lib/libiconv.so.2 (0x00007fe542e58000)
	libnl-route-3.so.200 => /usr/lib64/libnl-route-3.so.200 (0x00007fe542bea000)
	libnl-3.so.200 => /usr/lib64/libnl-3.so.200 (0x00007fe5429c9000)

check that helics shared library is using libraries from environments you really want

cd ../lib64
ldd libhelicsSharedLib.so
