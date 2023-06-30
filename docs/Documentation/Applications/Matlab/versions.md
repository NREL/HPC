# MATLAB Software Versions and Licenses

*Learn about the MATLAB software versions and licenses available for the NREL HPC systems.*

## Versions

The latest version available on NREL HPC systems is R2023a.

## Licenses

MATLAB is proprietary software. As such, users have access to a limited number
of licenses both for the base MATLAB software as well as some specialized
toolboxes.

To see which toolboxes are available, regardless of how they are licensed, start
an interactive MATLAB session and run:

```
>> ver
```

For a comprehensive list of available MATLAB-related licenses (including those not under active maintenance, such as the Database Toolbox), as
well as their current availability, run the following terminal command:

```
$ lmstat.matlab
```

Among other things, you should see the following:

```
Feature usage info:
  
Users of MATLAB: (Total of 6 licenses issued; Total of ... licenses in use)
  
Users of Compiler: (Total of 1 license issued; Total of ... licenses in use)
  
Users of Distrib_Computing_Toolbox: (Total of 4 licenses issued; Total of ... licenses in use)
  
Users of MATLAB_Distrib_Comp_Engine: (Total of 16 licenses issued; Total of ... licenses in use)
```

This documentation only covers the base MATLAB package and the Parallel
Computing Toolbox, which check out the "MATLAB" and "Distrib_Computing_Toolbox"
licenses, respectively.
