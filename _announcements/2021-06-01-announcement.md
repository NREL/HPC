---
title: June 2021 NREL HPC Monthly Update
data: 2021-06-01
layout: default
brief: FY22 Annual Call, CSC User & Apps Support, AUP Renewals
---


# Fiscal Year 2022 HPC Annual Call for Allocation Requests

The deadline for requests for HPC requests is next Monday, June 7, at Midnight Mountain Time. The submission portal will remain open after this date. 
However, late requests will receive lower priority than on-time requests. 

Please submit a request if you are a researcher at any national laboratory, university, or other organization pursing EERE-funded research, or if you 
are an NREL-affiliated researcher performing research aligned with the EERE mission funded through other organizations. Requests are welcome for 
current projects, projects where a funding request has been submitted, and projects where a funding request for FY21 is in preparation.

Additional information on the Eagle allocation process is available at our [Resource Allocation Requests](https://www.nrel.gov/hpc/resource-allocation-requests.html) page.
Please e-mail <hpc-requests@nrel.gov> if you have any additional questions.

# CSC User and Applications Support
A new Anaconda installation is in testing, and will be put into production shortly. To access the test installation, as always just enable the Test modules collection via

`module use /nopt/nrel/apps/modules/test/modulefiles`

and you should then see a conda/4.9.2 module.
   

Unlike with previous Anaconda installations, we have enabled the `conda activate` and `conda deactivate` syntax without requiring `conda init` (which 
creates "environmental" problems, something we're all against). The "source"ing syntax still works, but you now have the option to use either. 
Our hope is that enabling the conda commands will permit greater interoperability with scripts developed elsewhere (where conda activate may have 
worked), perhaps prove slightly faster, and eliminate awkward error messages.

This Conda deployment includes a new faster command for setting up new environments and installing packages. Instead of `conda ...` , 
you can try `mamba ...` , e.g., `mamba install tensorflow`.

Other application upgrades have been or will be deployed shortly. If they are not already in production, you may access the installations via the module use statement above.

|  App    | Version              |
|---------|----------------------|
| ANSYS   | 2021R1      |
| CMake   | 3.18.2      |
| COMSOL  | 5.6         |
| CUDA	  | 11, includes cudnn and development tools and libraries |
| GAMS	  | 34.3.0      |
| MATLAB  | R2020b      |
| MPT	  | 2.23        |
| OpenMPI | 4.1.0, including Java support |


# AUP Renewals
You may have received an email from DocuSign (dse_NA3@docusign.net) requesting that you renew your NREL HPC Appropriate Use Policy (AUP).  We 
are required to maintain these agreements and, should you receive one, it will be necessary for you to complete it within 30 days in order to 
continue accessing HPC systems.  We appreciate your cooperation.
 



