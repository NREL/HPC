# Environment Modules on Eagle

Eagle uses the Lmod environment modules system to easily manage software environments. Modules facilitate the use of different versions of applications, libraries, and toolchains, which enables support of multiple package versions concurrently.

Modules typically just set environment variables that one might traditionally do manually by, for example, adding export or setenv commands to their login script. Modules add the ability to back out changes in an orderly manner as well, so users can change their environment in a reversible way.

Modulefiles can be hosted in any location that the user makes known to the Lmod system via the ```module use``` command. By copying the reference modules, we set up into custom locations and modifying to your own tastes, you can host personal, project, or multi-project collections that you have complete control over.

Our production location for module examples is ```/nopt/nrel/apps/modules/default/modulefiles```. This path is set for everyone by default. In addition, motivated users who would like to support an application for the "energy community" on Eagle may request write access to the "e-com" space. Modules there are located in ```/nopt/nrel/ecom/modulefiles```, and are usable by all Eagle users (see the ```module use``` command under "Common Module Commands" in the accordion below).

??? abstract "Common Module Commands"
    The module command accepts parameters that enable users to inquire about and change the module environment. Most of the basic functionality can be accessed through the following commands.

    | Option | Description |
    | -------| ------------| 
    | spider | Prints available modules in a path-agnostic format.| 
    | avail | Prints available modules grouped by path. Note that in Eagle's layout, these two commands return roughly the same information| 
    | list | Prints all currently loaded modules.| 
    | display<br>'name' | Prints settings and paths specified for a particular module.| 
    | help 'name' | Prints help message for a particular module.| 
    | load 'name' | Loads particular module. For modules listed as the '(default)', the short package name is sufficient. To load another version of the package the long package name is required (e.g., ```module load fftw/3.3.8/gcc-7.3.0```).| 
    | unload 'name' | Unloads particular module.| 
    | swap <br> 'name 1'<br>'name 2' | First unload modName1 and then load modName2. | 
    | use {-a} <br> A_PATH | Prefix {suffix} the path $A_PATH to your $MODULEPATH variable, in order to find modules in that location.| 
    | unuse {-a} <br> A_PATH | Remove the path $A_PATH from your $MODULEPATH variable. | 

??? abstract "Module Organization on Eagle"
    The modulefiles that we provide are only a starting point. For maximum control, users should copy these files from the locations in /nopt to their own locations for which they have write access.

    Module files for baseline applications, libraries, frameworks, and toolchains are located in the ```/nopt/nrel/apps/modules/default/modulefiles directory.```
    Users may and should freely copy these example modulefiles to preferred locations and customize them for their own use cases. This can be particularly desirable to preserve a critical workflow as the software environment changes on Eagle, or to change the behavior, e.g., turn off automatic loading of prerequisites. In order to add a location to be searched regularly for available modules, the module use command may be added to a login script (e.g., ```.bash_profile```) or issued in an interactive shell or job script:
    ```
    module use -a /projects/{allocation}/modules/default/modulefiles
    module use -a /home/{username}/modules/default/modulefiles
    ```
    The -a flag appends the path that follows to environment variable MODULEPATH; leaving it out will prepend the path. The first module found in searching $MODULEPATH is used, so the search order is important.

    Since new versions of software are periodically added to the system, check current availability with the ```module spider``` command. If a module is needed often, the ```module load <module_name>``` command can be put in ```.bash_profile``` or other shell startup files.

??? abstract "Examples"
    To load a module:
    ```
    $ module load <module_name>/<version>
    ```
    Here ```<module_name>``` is to be replaced by the name of the module to load. It is advised to ALWAYS include the full versioning in your load statements, and not rely on explicit or implicit default behaviors.

    To get a list of available modules, type:

    ```
    $ module avail
    ```

    It's a good idea to look at two other commands to see what a module does, and what software dependencies there are, as illustrated below:

    ```
    [user@el4 04:05:26 ~]$ module show comp-intel/2018.0.3
    ...

    [user@el4 04:05:37 ~]$ module help comp-intel/2018.0.3

    ...
    ```
    The environment variables set by the module can then be used in build scripts. It is not necessary to load a module in order to use the ```module display``` command, this may be done at any time to see what a module does.

    Module files for different versions can easily be swapped:
    ```
    [user@el4 04:05:42 ~]$ module load openmpi/3.1.3/gcc-7.3.0
    [user@el4 04:06:52 ~]$ module list
    Currently Loaded Modulefiles:
    1) openmpi/3.1.3/gcc-7.3.0
    [user@el4 04:06:54 ~]$ module swap openmpi/3.1.3/gcc-7.3.0 openmpi/2.1.5/gcc-7.3.0
    [user@el4 04:07:09 ~]$ module list
    Currently Loaded Modulefiles:
    1) openmpi/2.1.5/gcc-7.3.0
    ```
??? abstract "Setting Up Personal and Project Modules from Existing Ones"
    ```
    mkdir -p $HOME/modules/default/modulefiles
    cd $HOME/modules/default/modulefiles
    mkdir comp-intel intel-mpi
    export TMP_PREFIX=/nopt/nrel/apps/modules/default/modulefiles
    cp $TMP_PREFIX/comp-intel/2018.0.3 comp-intel/.
    cp $TMP_PREFIX/intel-mpi/2018.0.3.lua intel-mpi/.
    cp $HOME/.bash_profile $HOME/.bash_profile.bak
    echo >> $HOME/.bash_profile
    echo "module use $HOME/modules/default/modulefiles" >> $HOME/.bash_profile
    ``` 
    Assuming you're using the bash shell, once you logout and log back in, you should see your new modules via ```module avail```. At that point, you are free to rename, edit, and configure as you see fit. For example, Intel compilers rely on a background GCC compiler in the environment. By default, the system version (4.8.5) is used, but you could add the ```gcc/7.3.0``` module to your collection, and create a comp-intel dependency on it so your build environment automatically uses the more modern GCC version.

    Of course, by changing ```$HOME``` in the instructions above to a project location (e.g., ```/projects/<your project name>```), you can create module collections that all users on a project can see and use.

    Finally, the ```modules/default/modulefile```s pattern is only convention—you can use any path that fits your needs.
