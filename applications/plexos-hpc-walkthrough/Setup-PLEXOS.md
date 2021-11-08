# Setting up PLEXOS for Eagle

The following file describes the initial setup needed to run PLEXOS on NREL's
HPC Eagle.

## Create a new PLEXOS License

Before we can run PLEXOS, we need to create a license file in eagle. For this,
run the following commands with some minor modifications

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
        <UserEmail val="$EMAIL" />       
  <CompanyCode val="6E-86-2D-7E-E2-FF-E9-1C-21-55-40-A0-45-40-A6-F0" />       
</XmlRegistryRoot>'   > ~/.config/PLEXOS/EE_reg.xml
```

Please note that you will need to modify the environment variable `EMAIL` to be
your own work email.

## Setup the PLEXOS Environment

### Setting up for the first time

1. We need to load a few modules and create the requisite conda environment. First,
we need to create a conda environment for PLEXOS.

  ```bash
  module purge
  module load conda
  conda create -n plex1 r-essentials
  ```

2. Log out and log back in. Load the following modules and activate the conda Environment

  ```bash
  module purge
  module load comp-intel intel-mpi mkl conda
  conda activate plex1
  ```

3. We need to install one, `rplexos` library from source. To do this, execute
the following commands

  ```bash
  mkdir /home/$USER/temporary    
  cd /home/$USER/temporary
  git clone https://github.com/NREL/rplexos.git
  cd rplexos
  CXX=`which icpc` R CMD INSTALL .
  ```

  *Note*: `rplexos` needs to beuilt using an intel compiler and R always wishes to
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

4. For some of the examples in [`Run-PLEXOS.md`](Run-PLEXOS.md), we need install
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

5. Finally make sure we have numpy and pandas in the `plex1` conda environment.

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


# Previous Documentation Version

Clone workshop from github to /scratch  :


clone github.com/gridmod/<workshop> use http

```bash
cd /scratch/$USER
git clone https://github.com/GridMod/MSPCM-Workshop.git
cd MSPCM-Workshop

```

Make a temporary directory for Plexos:

```bash
mkdir -p /scratch/$USER/tmp
```

Setup license file, EE_reg.xml, referencing HPC License in default location

```bash
rm -f ~/.config/EE_reg.xml
mkdir -p ~/.config/PLEXOS
cat > ~/.config/PLEXOS/EE_reg.xml << EOF
<?xml version="1.0"?>
<XmlRegistryRoot>
  <comms>
    <licServer_IP val="10.60.3.188" />
    <licServer_CommsPort val="399" />
    <LastLicTypeUsed val="server" />
  </comms>
  <UserName val="$USER" />
  <Company val="National Renewable Energy Lab" />
  <UserEmail val="example@example.com" />
  <CompanyCode val="6E-86-2D-7E-E2-FF-E9-1C-21-55-40-A0-45-40-A6-F0" />
</XmlRegistryRoot>
EOF
```
