---
title: Setting Up Plexos
parent: Plexos
grand_parent: Applications
---

## Loading the Appropriate Modules

!!! info
    A user can only run PLEXOS with Gurobi solvers on the clusters at this time. Please set up your model accordingly.

PLEXOS XML model files can only run with the Gurobi solver specified while creating the models. The most common combinations you may encounter are

| PLEXOS Module   | Gurobi Module |
|:---------------:|:-------------:|
| plexos/9.000R09 | gurobi/9.5.1  |
| plexos/9.200R05 | gurobi/10.0.1 |
| plexos/9.200R06 | gurobi/10.0.2 |

Please [contact us](https://www.nrel.gov/hpc/contact-us.html) if you encounter any issues or require a newer version.

## Setting up the License

Before we can run PLEXOS, we need to create a license file on the cluster. For this, run the following commands with some minor modifications

??? example "EE_reg.xml"

    ```bash
    mkdir -p ~/.config/PLEXOS
    echo '<?xml version="1.0"?>
    <XmlRegistryRoot>
      <comms>
        <licServer_IP val="10.60.3.188" />
        <licServer_CommsPort val="399" />
        <licServer_IP_Secondary />
        <connect>
          <PrimaryServer_Port />
          <SecondaryServer_Port />
        </connect>
        <licServer_CommsPort_Secondary />
        <LastLicTypeUsed val="server" />
      </comms>
      <server>
        <licServer_LogFolder val="/tmp/" />
        <licServer_LogEvents val="true" />
      </server>
      <proxy_cred>
        <proxy_ip val="" />
        <proxy_port val="" />
        <proxy_uname val="" />
        <proxy_pass val="" />
      </proxy_cred>
      <BannedList>
        <BanListedMachines val="true" />
      </BannedList>
      <ProductUpdates>
        <LastUpdateDate val="10/10/2021 13:11:10" />
      </ProductUpdates>
      <UserName />
      <Company />
      <UserEmail />
      <CompanyCode />
      <LicenseServerRequestCount />
    </XmlRegistryRoot>'   > ~/.config/PLEXOS/EE_reg.xml
    ```
## Optional: Conda environment for PLEXOS with Python and R

!!! note
    The following instructions are NOT required for only running PLEXOS. One only needs to load the relevant Gurobi and PLEXOS modules to run a PLEXOS XML database. Users may combine these runs with conda, Julia, or other software simply by loading the relevant modules and activating the appropriate conda and Julia environments.

1. We need to load a few modules and create the requisite conda environment. First, we need to create a conda environment for PLEXOS.
    ```bash
    module purge
    module load conda
    conda create -n plex1 r-essentials
    ```
2. Log out and log back in. Load the following modules and activate the conda environment
    ```bash
    module purge
    module load comp-intel intel-mpi mkl conda
    conda activate plex1
    ```
3. Install additional R libraries using conda
    ```bash
    conda install r-doParallel
    conda install r-RSQLite
    conda install r-testthat
    conda install r-covr
    ```

    !!! note
        Most of the R libraries should be added as part of the initial install, but keep an eye out for the following packages.

    !!! info
        See [below](setup_plexos.md#using-your-own-version-of-r-and-python) if you wish to use your own version of R and Python for PLEXOS.

4. We need to install one, `rplexos` library from source. To do this, execute the following commands
    ```bash
    mkdir /home/$USER/temporary    
    cd /home/$USER/temporary
    git clone https://github.com/NREL/rplexos.git
    cd rplexos
    CXX=`which icpc` R CMD INSTALL .
    ```

    !!! note
        `rplexos` needs to be built using an Intel compiler and R always wishes to build libraries using the same compilers that was used in its creation. If setting `CXX=which icpc` shown above does not work, we need to fool R by renaming the intel C++ compiler using a symbolic link. *This is a hack* and should only be used if the above way of installation fails. In order for the hack run the following after replacing username in the 3rd line with your own username.
        ```bash
        ln -s `which icpc` x86_64-conda_cos6-linux-gnu-c++
        export PATH=`pwd`:$PATH
        Rscript -e  "install.packages('/home/username/temporary/rplexos/',repos=NULL,type='source')"
        rm x86_64-conda_cos6-linux-gnu-c++
        ```

5. For some PLEXOS examples, we need to install an additional package called `plexos-coad`. For this run the following
    ```bash
    cd /scratch/$USER
    git clone https://github.com/Computational-Energy/plexos-coad.git
    cd plexos-coad

    #patch for python 3.9
    tofix=`grep -lr getchild`
    for f in $tofix ; do sed -i3x "s/for el_data in elem.getchildren()/for el_data in list\(elem\)/" $f ; done
    pip install Cython
    python setup.py install
    ```

6. Finally make sure we have numpy and pandas in the `plex1` conda environment.
    ```bash
    pip install numpy pandas
    ```

### Loading an existing PLEXOS environment

If you have successfully followed all the instructions in the previous subsection and installed PLEXOS, you can simply load the following modules and activate the conda environment

```bash
module purge
module load comp-intel intel-mpi mkl conda
conda activate plex1
```

### Using your own version of R and Python

This section is in regards to Point 3 in [setting up the PLEXOS environment](#conda-environment-for-plexos-with-python-and-r).
The following R libraries will need to be installed manually in this case.

```
install.packages("data.table")
install.packages("DBI")
install.packages("dbplyr")
install.packages("doParallel")
install.packages("dplyr")
install.packages("foreach")
install.packages("lubridate")
install.packages("magrittr")
install.packages("parallel")
install.packages("Rcpp")
install.packages("RSQLite")
install.packages("stringi")
install.packages("tidyr")
install.packages("knitr")
install.packages("testthat")
install.packages("ggplot2")
install.packages("covr")
install.packages("tidyverse")
```

After installing the above, follow the remainder of the installation starting with
point 4.