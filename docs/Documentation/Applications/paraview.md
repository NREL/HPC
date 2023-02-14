#Using ParaView Software on the Eagle System
Learn how to use ParaView software—an open-source, multi-platform data analysis and visualization application—on the Eagle system.

##Introduction
ParaView is an open-source, multi-platform data analysis and visualization application. ParaView users can quickly build visualizations to analyze their data using qualitative and quantitative techniques. The data exploration can be done interactively in 3D or programmatically using ParaView's batch processing capabilities. ParaView was developed to analyze extremely large data sets using distributed memory computing resources. It can be run on supercomputers to analyze data sets of terascale as well as on laptops for smaller data.

##Remote Visualization
ParaView can be run as a standalone application on a desktop or remotely from the Redeye or Eagle visualization nodes. ParaView can also be run as a client-server application. The ParaView client will run on your desktop while the server will run at the remote high-performance computing (HPC) resource. The following describes the steps you will take to install ParaView on your desktop and configure it so that you can launch remote-parallel jobs on NREL systems from within the ParaView GUI. Running ParaView remotely in a client-server configuration involves establishing an SSH (secure shell) tunnel to the login node, launching the ParaView server, connecting the server to the client over the tunnel via a socket. For a detailed walkthrough of these steps, see the NREL HPC GitHub repository.

###Install the ParaView Client
The first step is to install ParaView. It is recommended that you use the binaries provided by Kitware on your workstation matching the NREL installed version. This ensures client-server compatibility. The version number that you install must identically match the version installed at NREL. To download the correct ParaView client binary version for your desktop environment, visit the ParaView website.

###Install Dependencies
In order to use ParaView remotely, you will also need an SSH client and a terminal program. On Linux and Mac these functions are provided by the SSH and xterm programs, both of which should come pre-installed on most Linux and Mac systems. On Windows the SSH and terminal functions are provided by the programs plink.exe and cmd.exe, of which only cmd.exe will come pre-installed. The SSH client, plink.exe, needs to be installed before using ParaView.

##Immersive Visualization
ParaView is supported in the Insight Center's immersive virtual environment. i
[Learn about the Insight Center](https://www.nrel.gov/computational-science/insight-center.html). 

For assistance, contact [Kenny Gruchalla](Kenny.Gruchalla@nrel.gov).
