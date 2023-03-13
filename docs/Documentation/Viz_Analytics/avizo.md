# Using Avizo Software 
*Learn about using Avizo 3D analysis software for scientific and industrial data on the Eagle System.*

## Introduction
Avizo software is a powerful, multifaceted commercial software tool for visualizing, manipulating, and understanding scientific and industrial data. 
NREL licenses a limited number of Avizo tokens, allowing a small number of Avizo instances to be run lab-wide.

## Running Remotely
Avizo is installed and can be run remotely from the Eagle visualization node. 
First, launch a TurboVNC remote desktop. 
Then from a terminal in that remote desktop:

``` bash
% module load avizo 
% vglrun avizo
```

##Running Locally
Avizo can also be run on a local desktop connected to the NREL network — the machine must be connected to the network to access the license server.

## Install the Software
First install the software:

[Avizo 9.3 for Windows](ftp://ftp.vsg3d.com/private/MASTERS/Avizo/9.3.0/f93abe0f/Avizo-930-Windows64-VC12.exe)

[Avizo 9.3 for Linux](ftp://ftp.vsg3d.com/private/MASTERS/Avizo/9.3.0/f93abe0f/Avizo-930-Linux64-gcc44.bin)

[Avizo 9.3 for Mac OS X](ftp://ftp.vsg3d.com/private/MASTERS/Avizo/9.3.0/f93abe0f/Avizo-930-MacOSX-gcc42.pkg)

## Activate the License
Use FNP license server:

`SERVER license-1.hpc.nrel.gov:27003`
