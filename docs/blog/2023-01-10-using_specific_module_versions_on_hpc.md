# Using Specific Module Versions on the HPC

Modules on NREL HPCs are updated to a newer versions with on a regular basis.
Since Lmod, the undelying module system, sets the most recent version of a module as the default, a user's typical workflow may break if they are not specifying the exact module version in their scripts.

For example, at the time of this writing, the current default module for Conda on Eagle is 4.9.2.
If a user wishes to use conda v4.12.0, they must specify the version in the module command as

```bash
module load conda/4.12.0
```

The user can also create custom module files for their own use and point to them.
For example, assuming a user has custom TCL or LUA module files in the following directory

```bash
/home/${USER}/private_modules/
```

They can use these module files by adding it to the module search path using the following command

```bash
module use -a /home/${USER}/private_modules/
```

Furthermore, if a user wishes to have these module paths available at all times, they can update their `.bash_profile` or `.bashrc` file in their home directory. For example by using the following command

```bash
echo 'module use -a /home/${USER}/private_modules/' >> /home/${USER}/.bash_profile
```

or a text editor.
As a quick reminder, `.bash_profile` and `.bashrc` are simply configuration files that enable the user to customize your Linux or MacOS terminal experience, assuming they are using Bash as their shell (Similar to Command Prompt on Windows).
The difference between `.bash_profile` and `.bashrc` is that the former is executed for login shells while the latter in executed for non-login shells, e.g., `.bash_profile` is executed when a user SSHs into Eagle, whereas `.bashrc` is executed when one opens a new terminal.