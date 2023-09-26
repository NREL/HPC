# Setting up PLEXOS on Eagle

The following file contains information about setting up PLEXOS for the first time on your account.

## Setting up the license file

Before we can run PLEXOS, we need to create a license file in Eagle. 
For this, run the following commands with some minor modifications

### License file for Versions 9.0+

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

### License file for Versions < 9.0

```bash
mkdir -p ~/.config/PLEXOS
export EMAIL=your.email@nrel.gov
echo '<?xml version="1.0"?>       
<XmlRegistryRoot>       
  <comms>       
    <licServer_IP val="10.60.3.188" />       
    <licServer_CommsPort val="399" />       
    <LastLicTypeUsed val="server" />       
  </comms>       
        <UserName val="${USER}" />       
  <Company val="National Renewable Energy Lab" />       
        <UserEmail val="${EMAIL}" />       
  <CompanyCode val="6E-86-2D-7E-E2-FF-E9-1C-21-55-40-A0-45-40-A6-F0" />       
</XmlRegistryRoot>'   > ~/.config/PLEXOS/EE_reg.xml
```

Please note that you will need to modify the environment variable `EMAIL` to be
your own work email.

## Setup the PLEXOS Conda Environment

### Setting up for the first time

1. We need to load a few modules and create the requisite conda environment. First,
we need to create a conda environment for PLEXOS.

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

  ```
  conda install r-doParallel
  conda install r-RSQLite
  conda install r-testthat
  conda install r-covr
  ```

  *Note* Most of the R libraries should be added as part of the initial install, but
  keep an eye out for the following packages

  ```
  conda install r-data.table r-DBI r-dbplyr r-dplyr r-foreach r-lubridate r-magrittr       
  conda install r-parallel r-Rcpp r-stringi r-tidyr r-knitr r-ggplot2 r-tidyverse       
  ```

  *Note* See [below](Setup-PLEXOS.md#using-your-own-version-of-r-and-python) if you wish to use your own version of R and Python for PLEXOS

4. We need to install one, `rplexos` library from source. To do this, execute
the following commands

  ```bash
  mkdir /home/$USER/temporary    
  cd /home/$USER/temporary
  git clone https://github.com/NREL/rplexos.git
  cd rplexos
  CXX=`which icpc` R CMD INSTALL .
  ```

  *Note*: `rplexos` needs to be built using an Intel compiler and R always wishes to
  build libraries using the same compilers that was used in its creation. If
  setting `CXX=which icpc` shown above does not work, we need to fool R by renaming
  the intel C++ compiler using a sym-link. *This is a hack* and should only be used
  if the above way of installation fails. In order for the hack run the following
  after replacing `username` in the 3rd line with your own username.

  ```bash
  ln -s `which icpc` x86_64-conda_cos6-linux-gnu-c++
  export PATH=`pwd`:$PATH
  Rscript -e  "install.packages('/home/username/temporary/rplexos/',repos=NULL,type='source')"
  rm x86_64-conda_cos6-linux-gnu-c++
  ```

5. For some of the examples in [`Run-PLEXOS.md`](Run-PLEXOS.md), we need install
an additional package called `plexos-coad`. For this run the following

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

If you have successfully followed all the instructions in the previous subsection
and installed PLEXOS, you can simply load the following modules and activate the
conda environment

```bash
module purge
module load comp-intel intel-mpi mkl conda
conda activate plex1
```

### Using your own version of R and Python

This section is in regards to Point 3 in [setting up the PLEXOS environment](Setup-PLEXOS.md#setting-up-for-the-first-time).
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
