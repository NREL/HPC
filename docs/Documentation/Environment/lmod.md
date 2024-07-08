---
layout: default
title: Modules
---
# Environment Modules

The Lmod environment modules system is used to easily manage software environments. Modules facilitate the use of different versions of applications, libraries, and toolchains, which enables support of multiple package versions concurrently.

Modules typically just set environment variables that one might traditionally do manually by, for example, adding export or setenv commands to their login script. Modules add the ability to back out changes in an orderly manner as well, so users can change their environment in a reversible way. To learn how to build your own modules see [Building an Application](/Documentation/Environment/Building_Packages/modules/).

## Common Module Commands
The ```module``` command accepts parameters that enable users to inquire about and change the module environment. Most of the basic functionality can be accessed through the following commands.

| Option | Description |
| -------| ------------| 
| spider | Prints available modules in a path-agnostic format.| 
| avail | Prints available modules grouped by path.| 
| list | Prints all currently loaded modules.| 
| display<br>'name' | Prints settings and paths specified for a particular module.| 
| help 'name' | Prints help message for a particular module.| 
| load 'name' | Loads particular module. For modules listed as the '(default)', the short package name is sufficient. To load another version of the package the long package name is required (e.g., ```module load fftw/3.3.8/gcc-7.3.0```).| 
| unload 'name' | Unloads particular module.| 
| swap <br> 'name 1'<br>'name 2' | First unload modName1 and then load modName2. | 
| use {-a} <br> A_PATH | Prefix {suffix} the path $A_PATH to your $MODULEPATH variable, in order to find modules in that location.| 
| unuse {-a} <br> A_PATH | Remove the path $A_PATH from your $MODULEPATH variable. | 

## Examples
### Determining loaded modules
To determine which modules are already loaded into your system, run the command:
```
$ module list
```

On Kestrel, the following 10 modules are loaded automatically:
```
Currently Loaded Modules:
  1) craype-x86-spr       4) perftools-base/23.12.0   7) cray-dsmml/0.2.2     10) PrgEnv-cray/8.5.0
  2) libfabric/1.15.2.0   5) cce/17.0.0               8) cray-mpich/8.1.28
  3) craype-network-ofi   6) craype/2.7.30            9) cray-libsci/23.12.5

```
    
### Loading and unloading a module
```
$ module load <module_name>/<version>
...

$ module unload <module_name>/<version>
...
```
Here ```<module_name>``` is to be replaced by the name of the module to load. It is advised to ALWAYS include the full versioning in your load statements, and not rely on explicit or implicit default behaviors.

### Seeing available modules
To get a list of available modules, type:

```
$ module avail
```

This should outut a full list of all modules and their versions in the system available for you to load. The modules denoted with *(L)* are already loaded in your system. The module versions denoted with *(D)* are the default versions that will load if you do not specify the version when running ```$ module load```.

To get a list of the available module *defaults*, type:
```
$ module --default avail
```

### Seeing module specifics
It's a good idea to look at two other commands to see what a module does, and what software dependencies there are, as illustrated below:

```
$ module show comp-intel/2018.0.3
...

$ module help comp-intel/2018.0.3
...
```

The environment variables set by the module can then be used in build scripts.

It is not necessary to load a module in order to use the ```module display``` command, this may be done at any time to see what a module does.

### Swap a module environment
Module files for different versions can easily be swapped:
```
$ module load openmpi/3.1.3/gcc-7.3.0
$ module list
Currently Loaded Modulefiles:
1) openmpi/3.1.3/gcc-7.3.0
$ module swap openmpi/3.1.3/gcc-7.3.0 openmpi/2.1.5/gcc-7.3.0
$ module list
Currently Loaded Modulefiles:
1) openmpi/2.1.5/gcc-7.3.0
```

### Create your own modules
An in depth example on how to do this can be found at [Building an Application](/Documentation/Environment/Building_Packages/modules/).
