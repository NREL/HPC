## Introduction

Spack is an HPC-centric package manager for acquiring, building, and managing HPC applications as well as all their dependencies, down to the compilers themselves. Like frameworks such as Anaconda, it is associated with a repository of both source-code and binary packages. Builds are fully configurable through a DSL at the command line as well as in YAML files. Maintaining many build-time permutations of packages is simple through an automatic and user-transparent hashing mechanism. The Spack system also automatically creates (customizable) environment modulefiles for each built package.

## Installation

Multiple installations of Spack can easily be kept, and each is separate from the others by virtue of the environment variable `SPACK_ROOT`. 
All package, build, and modulefile content is kept inside the `SPACK_ROOT` path, so working with different package collections is as simple as setting `SPACK_ROOT` to the appropriate location. 
The only exception to this orthogonality are `YAML` files in `$HOME/.spack/<platform>`.
Installing a Spack instance is as easy as

`git clone https://github.com/spack/spack.git`

Once the initial Spack instance is set up, it is easy to create new ones from it through

`spack clone <new_path>`

`SPACK_ROOT` will need to point to `<new_path>` in order to be consistent.

Spack environment setup can be done by sourcing `$SPACK_ROOT/share/spack/setup-env.sh`, or by simply adding `$SPACK_ROOT/bin` to your PATH. 

`source $SPACK_ROOT/share/spack/setup-env.sh`
or 
`export PATH=$SPACK_ROOT/bin:$PATH`



## Setting Up Compilers

Spack is able to find certain compilers on its own, and will add them to your environment as it does. 
In order to obtain the list of available compilers on Eagle the user can run `module avail`, the user can then load the compiler of interest using `module use <compiler>`.
To see which compilers your Spack collections know about, type

`spack compilers`

To add an existing compiler installation to your collection, point Spack to its location through

`spack add compiler <path to Spack-installed compiler directory with hash in name>`

The command will add to `$HOME/.spack/linux/compilers.yaml`. 
To configure more generally, move changes to one of the lower-precedence `compilers.yaml` files (paths described below in Configuration section).
Spack has enough facility with standard compilers (e.g., GCC, Intel, PGI, Clang) that this should be all that’s required to use the added compiler successfully.

## Available Packages in Repo

|<div style="width:198px">Command</div>                         |Description                                                                                                     |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------|
|`spack list                   ` |all available packages by name. Dumps repo content, so if use local repo, this should dump local package load.  |
|`spack list <pattern>         ` |all available packages that have `<pattern>` somewhere in their name. `<pattern>` is simple, not regex.         |
|`spack info <package_name>    ` | available versions classified as safe, preferred, or variants, as well as dependencies. Variants are important for selecting certain build features, e.g., with/without Infiniband support.| 
|<div style="width:198px">`spack versions <package_name>`</div> | see which versions are available                                                                               | 



## Installed packages

|<div style="width:98px">Command</div>                        |Description                                                                          |
|----------------------------------|-------------------------------------------------------------------------------------|
|`spack find	                 ` |list all locally installed packages                                                  | 
|`spack find --deps <package> `    |list dependencies of `<package>`                                                     |
|`spack find --explicit	     `     |list packages that were explicitly requested via spack install                       | 
|`spack find --implicit	     `     |list packages that were installed as a dependency to an explicitly installed package |
|`spack find --long	         `     |include partial hash in package listing. Useful to see distinct builds               | 
|`spack find --paths	         ` |show installation paths                                                              |


Finding how an installed package was built does not seem as straightforward as it should be. 
Probably the best way is to examine `<install_path>/.spack/build.env`, where `<install_path>` is the Spack-created directory with the hash for the package being queried. 
The environment variable `SPACK_SHORT_SPEC` in `build.env` contains the Spack command that can be used to recreate the package (including any implicitly defined variables, e.g., arch). 
The 7-character short hash is also included, and should be excluded from any spack install command.


|<div style="width:98px">Symbols</div>       | Description |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
|`@	         ` |package versions. Can use range operator “:”, e.g., X@1.2:1.4 . Range is inclusive and open-ended, e.g., “X@1.4:” matches any version of package X 1.4 or higher.| 
|`%	         ` |compiler spec. Can include versioning, e.g., X%gcc@4.8.5 |
|`+,-,~       `| build options. +opt, -opt, “~” is equivalent to “-“ |
|`name=value`  |build options for non-Boolean flags. Special names are cflags, cxxflags, fflags, cppflags, ldflags, and ldlibs |
|`target=value`|for defined CPU architectures, e.g., target=haswell    |      
|`os=value	 ` |for defined operating systems                           |         
|`^	         ` |dependency specification, using above specs as appropriate|       
|`^/<hash>	 ` |specify dependency where `<hash>` is of sufficient length to resolve uniquely |

## External Packages

Sometimes dependencies are expected to be resolved through a package that is installed as part of the host system, or otherwise outside of the Spack database. 
One example is Slurm integration into MPI builds. 
If you were to try to add a dependency on one of the listed Slurms in the Spack database, you might see, e.g.,

```bash
[$user@el2 ~]$ spack spec openmpi@3.1.3%gcc@7.3.0 ^slurm@19-05-3-2
Input spec
--------------------------------
openmpi@3.1.3%gcc@7.3.0
    ^slurm@19-05-3-2

Concretized
--------------------------------
==> Error: The spec 'slurm' is configured as not buildable, and no matching external installs were found
```

Given that something like Slurm is integrated deeply into the runtime infrastructure of our local environment, we really want to point to the local installation. 
The way to do that is with a `packages.yaml` file, which can reside in the standard Spack locations (see Configuration below). 
See the Spack [docs](https://spack.readthedocs.io/en/latest/) on external packages for more detail. 
In the above example at time of writing, we would like to build OpenMPI against our installed `Slurm 19.05.2`. 
So, you can create file `~/.spack/linux/packages.yaml` with the contents

```yaml
packages:
  slurm:
    paths:
      slurm@18-08-0-3: /nopt/slurm/18.08.3
      slurm@19-05-0-2: /nopt/slurm/19.05.2
```

that will enable builds against both installed Slurm versions. 
Then you should see

```bash
[$user@el2 ~]$ spack spec openmpi@3.1.3%gcc@7.3.0 ^slurm@19-05-0-2
Input spec
--------------------------------
openmpi@3.1.3%gcc@7.3.0
    ^slurm@19-05-0-2

Concretized
--------------------------------
openmpi@3.1.3%gcc@7.3.0 cflags="-O2 -march=skylake-avx512 -mtune=skylake-avx512" cxxflags="-O2 -march=skylake-avx512 -mtune=skylake-avx512" fflags="-O2 -march=skylake-avx512 -mtune=skylake-avx512" +cuda+cxx_exceptions fabrics=verbs ~java~legacylaunchers~memchecker+pmi schedulers=slurm ~sqlite3~thread_multiple+vt arch=linux-centos7-x86_64
-
    ^slurm@19-05-0-2%gcc@7.3.0 cflags="-O2 -march=skylake-avx512 -mtune=skylake-avx512" cxxflags="-O2 -march=skylake-avx512 -mtune=skylake-avx512" fflags="-O2 -march=skylake-avx512 -mtune=skylake-avx512" ~gtk~hdf5~hwloc~mariadb+readline arch=linux-centos7-x86_64
```

where the Slurm dependency will be satisfied with the installed Slurm (cflags, cxxflags, and arch are coming from site-wide configuration in `/nopt/nrel/apps/base/2018-12-02/spack/etc/spack/compilers.yaml`; the variants string is likely coming from the configuration in the Spack database, and should be ignored).

## Virtual Packages 

It is possible to specify some packages for which multiple options are available at a higher level. 
For example, `mpi` is a virtual package specifier that can resolve to mpich, openmpi, Intel MPI, etc. 
If a package's dependencies are spec'd in terms of a virtual package, Spack will choose a specific package at build time according to site preferences.
Choices can be constrained by spec, e.g.,

`spack install X ^mpich@3`

would satisfy package X’s mpi dependency with some version 3 of MPICH.
You can see available providers of a virtual package with

`spack providers <vpackage>`

## Extensions

In many cases, frameworks have sub-package installations in standard locations within their own installations. 
A familiar example of this is Python and its usual module location in `lib(64)/python<version>/site-packages`, and pointed to via the environment variable `PYTHONPATH`.

To find available extensions

`spack extensions <package>`

Extensions are just packages, but they are not enabled for use out of the box. To do so (e.g., so that you could load the Python module after installing), you can either load the extension package’s environment module, or

`spack use <extension package>`

This only lasts for the current session, and is not of general interest. A more persistent option is to activate the extension:

`spack activate <extension package>`

This takes care of dependencies as well. The inverse operation is deactivation.


|<div style="width:98px">Command</div>| Description |
|-|-|
|`spack deactivate <extension package>`	        |deactivates extension alone. Will not deactivate if dependents exist |
|`spack deactivate --force <extension package>`	|deactivates regardless of dependents  |
|`spack deactivate --all <extension package>`	|    deactivates extension and all dependencies | 
|`spack deactivate --all <parent>`	            |deactivates all extensions of parent (e.g., `<python>`) | 


## Modules

Spack can auto-create environment modulefiles for the packages that it builds, both in Tcl for “environment modules” per se, and in Lua for Lmod. 
Auto-creation includes each dependency and option permutation, which can lead to excessive quantities of modulefiles. 
Spack also uses the package hash as part of the modulefile name, which can be somewhat disconcerting to users. 
These default behaviors can be treated in the active modules.yaml file, as well as practices used for support.
Tcl modulefiles are created in `$SPACK_ROOT/share/spack/modules` by default, and the equivalent Lmod location is `$SPACK_ROOT/share/spack/lmod`. 
Only Tcl modules are created by default. 
You can modify the active modules.yaml file in the following ways to affect some example behaviors:

#### To turn Lmod module creation on:

```
modules:
    enable:
        - tcl
        - lmod 
```

#### To change the modulefile naming pattern:

```
modules:
    tcl:
        naming_scheme: ‘{name}/{version}/{compiler.name}-{compiler.version}
```

would achieve the Eagle naming scheme. 
#### To remove default variable settings in the modulefile, e.g., CPATH:

```
modules:
    tcl:
        all:
            filter:
                environment_blacklist: [‘CPATH’]
```

Note that this would affect Tcl modulefiles only; if Spack also creates Lmod files, those would still contain default CPATH modification behavior.

#### To prevent certain modulefiles from being built, you can whitelist and blacklist:

```
modules:
    tcl:
        whitelist: [‘gcc’]
        blacklist: [‘%gcc@4.8.5’]
```

This would create modules for all versions of GCC built using the system compiler, but not for the system compiler itself.
There are a great many further behaviors that can be changed, see https://spack.readthedocs.io/en/latest/module_file_support.html#modules for more.

For general user support, it is not a bad idea to keep the modules that are publicly visible separate from the collection that Spack auto-generates. This involves some manual copying, but is generally not onerous as all rpaths are included in Spack-built binaries (i.e., you don’t have to worry about satisfying library dependencies for Spack applications with an auto-built module, since library paths are hard-coded into the application binaries). This separation also frees one from accepting Spack’s verbose coding formats within modulefiles, should you decide to maintain certain modulefiles another way.

## Configuration

Spack uses hierarchical customization files. 
Every package is a Python class, and inherits from the top-level class Package. 
Depending on the degree of site customization, you may want to fork the Spack repo to create your own customized Spack package.
There are 4 levels of configuration. In order of increasing precedence,

1.	Default: `$SPACK_ROOT/etc/spack/default`
2.	System-wide: `/etc/spack`
3.	Site-wide: `$SPACK_ROOT/etc/spack`
4.	User-specific: `$HOME/.spack`

Spack configuration uses YAML files, a subset of JSON native to Python.
There are 5 main configuration files.

1.	`compilers.yaml`. Customizations to the Spack-known compilers for all builds
    
    i.	Use full path to compilers
    
    ii.	Additional rpaths beyond the Spack repo
    
    iii.	Additional modules necessary when invoking compilers
    
    iv.	Mixing toolchains
    
    v.	Optimization flags
    
    vi.	Environment modifications

2.	`config.yaml`. Base functionality of Spack itself
    
    i.	install_tree: where to install packages
    
    ii.	build_stage: where to do compiles. For performance, can specify a local SSD or a RAMFS.
    
    iii.	modules_roots: where to install modulefiles

3.	`modules.yaml`. How to create modulefiles

    i.	whitelist/blacklist packages from having their own modulefiles created
    
    ii.	adjust hierarchies

4.	`packages.yaml`. Specific optimizations, such as multiple hardware targets.

    i.	dependencies, e.g., don’t build OpenSSL (usually want sysadmins to handle updates, etc.)

    ii.	mark specific packages as non-buildable, e.g., vendor MPIs
    
    iii.	preferences, e.g., BLAS -> MKL, LAPACK -> MKL

5.	`repos.yaml`
    
    i.	Directory-housed, not remote
    
    ii.	Specify other package locations
    
    iii.	Can then spec build in other configs (e.g., binary, don’t build)
    
    iv.	Precedence in YAML file order, but follows Spack precedence order (user > site > system > default)

### Variants: standard adjustments to package build
`spack edit …  `-- opens Python file for package, can easily write new variants

### Providers
`spack providers` -- virtual packages, e.g., blas, mpi, etc. Standards, not implementations. Abstraction of an implementation (blas/mkl, mpi/mpich, etc.)

### Mirrors
- mirrors.yaml: where packages are kept
- A repo is where build information is kept; a mirror is where code lives

```
MirrorTopLevel
	package_a
		package_a-version1.tar.gz
		package_a-version2.tar.gz
	package_b
		⋮
```

`spack mirror` to manage mirrors

### Repos
- Can take precedence from, e.g., a site repo
- Can namespace

```
packages
	repo.yaml
	alpha
		hotfix-patch-ABC.patch
		package.py
		package.pyc
	beta
	theta
```

