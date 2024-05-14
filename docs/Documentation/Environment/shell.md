# Shell Startup

*When you login to a linux based machine you interact with the operating system via a program called a shell.  There are various types of shell programs.  One of the more common is **bash**.  Bash is the default shell on NREL's HPC platforms.  This document describes ways you can customize your shell's, in particular, bash's behavior.*



## Getting Started
When you have a window open attached to a platform you are actually running a program on the remote computer, called a shell.  There are various types of shell programs.  One of the more common is *bash*.  

The shell program provides your link to the machine's operating system (OS).  It is the interface between a user and the computer. It controls the computer and provides output to the user.  There are various types of interfaces but here we discuss the command line interface. That is, you type commands and the computer responds.

### What happens on login

When you login to a machine you are put in your home directory.  You can see this by running the command **pwd**.  Run the command **ls -a** to get a listing of the files.  The **-a** option for the ls commands enables it to show files that are normally hidden.  You'll see two important files that are used for setting up your environment.

* .bash_profile
* .bashrc


These files are added to your home directory when your account is created.  

When you login the file .bash_profile is sourced (run) to set up your environment.  The environment includes settings for important variables, command aliases, and functions.  

Here is the default version of .bash_profile.  

```
[nreluser@el3 ~]$ cat ~/.bash_profile
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH
```

We'll discuss this file starting at the bottom. The environmental variable PATH is set.  PATH points to directories where the computer will look for commands to run.  You can append directories as show here.  The "new" PATH will be the PATH set at the system level plus the directories $HOME/.local/bin and $HOME/bin where $HOME is your home directory.

Notice the lines

```
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
```

The "if" statement says that if you have a file .bashrc in your home directory then run it.  The dot is shorthand for "source" and ~/  is shorthand for your home directory.

So lets look at the default ~/.bashrc file


```
[nreluser@el3 ~]$ cat /etc/skel/.bashrc
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
```

This just runs the system version of bashrc.  

Note in both of these files we have a place where you are encouraged to add user defined aliases and functions.  You can also set environmental variables, such as PATH and a related variable LD_LIBRARY_PATH.  You may want to load modules which also set environmental variables. 


### Suggestions (Philosophy)
We're going to discuss customizing your environment. This is done by editing these two files. Before we do that here are three suggestions.  

1. 	If you are new to linux use the **nano** editor
1. 	Make a backup of your current versions of the two files
1. 	Make additions in external files

Nano is an easy to learn and use text editor.  The official web page for nano is [https://www.nano-editor.org](https://www.nano-editor.org).  There are many on line tutorials.  There are other editors available but nano is a good starting point.

It is very easy to make mistakes when doing edits or you just might want to go back to a previous version.  So back it up.  Here are commands to do so.

```

[hpcuser2@eyas1 ~]$ NOW=`date +"%y%m%d%H%M"`
[hpcuser2@eyas1 ~]$ echo $NOW
2303221513
[hpcuser2@eyas1 ~]$ cp .bashrc bashrc.$NOW
[hpcuser2@eyas1 ~]$ cp .bash_profile bash_profile.$NOW
```
The first command creates a date/time stamp.  The last commands copy files using the date/time stamp as part of the filename.

```

[hpcuser2@eyas1 ~]$ ls *2303221513
bash_profile.2303221513  bashrc.2303221513
[hpcuser2@eyas1 ~]$ 
```

In most cases you won't need to edit both .bashrc and .bash_profile.  Since running .bash_profile runs .bashrc you can usually just edit .bashrc.  (See the section *Difference between login and interactive shells* which describes cases where .bashrc is run even if .bash_profile is not.)

Instead of adding a bunch of text to .bashrc make your additions in an external file(s) and just source those files inside of .bashrc.  The you can "turn off" additions by just commenting out the source lines.  Also, you can test additions by sourcing the file from the command lines.

### Additions

The most common additions to your environment fall into these categories:

1. Setting variables 
1. Creating Aliases
1. Loading modules
1. Adding Functions

We'll discuss each. We're going to assume that you created a directory **~/MYENV** and in that directory you have the files:

* myvars
* myaliases
* mymods
* myfuncs

Then to enable all of your additions you can add the following lines to your .bashrc file

```
if [ -f ~/MYENV/myvars ];    then . ~/MYENV/myvars ;    fi
if [ -f ~/MYENV/myaliases ]; then . ~/MYENV/myaliases ; fi
if [ -f ~/MYENV/mymods ];    then . ~/MYENV/mymods ;    fi
if [ -f ~/MYENV/myfuncs ];   then . ~/MYENV/myfuncs ;   fi
```

Note the additions will not take effect until you logout/login or until you run the command **source ~/.bashrc**   Before going through the logout/login process you should test your additions by manually running these commands in the terminal window.

#### Setting variables
We have discussed the PATH variable.  It points to directories which contain programs.  If you have an application that you built, say myapp in /projects/mystuff/apps you can add the line 

export PATH=/projects/mystuff/apps:$PATH

to your ~/MYENV/myvars file.  Then when you login the system will be able to find your application.  The directories in path variables are seperated by a ":". If you forget to add $PATH to the export line the new PATH variable will be truncated and you will not see many "system" commands.  

Another important variable is LD_LIBRARY_PATH.  This points to directories containing libraries your applications need that are not "bundled" with your code.  Assuming the libraries are in projects/mystuff/lib you would add the following line:

export LD_LIBRARY_PATH=/projects/mystuff/lib:$LD_LIBRARY_PATH


If you have a commercial application that requires a license server you may need to set a variable to point to it.  For example

export LSERVER=license-1.hpc.nrel.gov:4691



#### Creating aliases
Aliases are command short cuts.  If there is a complicated command that you often you might want to crate an alias for it.  You can get a list of aliases defined for you by just running the command alias.  The syntax for an alias is:

alias NAME="what you want to do"

Here are a few examples that you could add to your ~/MYENV/myalias file.

```
#Show my running and queued jobs in useful format
alias sq='squeue -u $USER --format='\''%10A%15l%15L%6D%20S%15P%15r%20V%N'\'''

#Kill all my running and queued jobs
alias killjobs="scancel -u $USER"

#Get a list of available modules
alias ma='module avail'

#Get the "source" for a git repository
alias git-home='git remote show origin'

#Get a compact list of loaded modules
alias mlist='module list 2>&1 |  egrep -v "Current|No modules loaded" | sed "s/..)//g"'

```


#### Loading modules
Most HPC platforms run module systems.  When you load a module changes some environmental variable setting.  Often PATH and LD_LIBARAY_PATH are changed.  In general loading a module will allow you to use a particular application or library.  

If you always want gcc version 12 and python 3.10 in you path then you could add the following to your ~/MYENV/mymods file

```
module load gcc/12.1.0  
module load python/3.10.2
```

Running the command **module avail** will show the modules installed on the system.

If you have modules that you created you can make them available to the load command by adding a command like the following in your ~/MYENV/mymods file.

module use /projects//mystuff/mods

The "module use" command needs to be before any module load command that loads your coustom modules.



#### Adding functions
Functions are like aliases but in general multiline and more complex. You can run the command **compgen -A function ** to see a list of defined functions.  Here are a few functions you might want to add to your environment

```
# given a name of a function or alias show its definition
func () 
{ 
    typeset -f $1 || alias $1
}

# find files in a directory that changed today
today () 
{ 
    local now=`date +"%Y-%m-%d"`;
    if (( $# > 0 )); then
        if [[ $1 == "-f" ]]; then
            find . -type f -newermt $now;
        fi;
        if [[ $1 == "-d" ]]; then
            find . -type d -newermt $now;
        fi;
    else
        find . -newermt $now;
    fi
}

```

Most people who have worked in HPC for some time have collected many functions and alias they would be willing to share with you.


If you have a number of files in your ~/MYENV directory you want sourced at startup you can replace the  set of 4 "if" lines shown above with a "for list" statemnet.  The following will source every file in the directory.  It will not source files in subdirectories within ~/MYENV.  If you want to temporarly turn off additions you can put them in a subdirectory ~/MYENV/OFF.  The find command shown here will return a list of files in the directory but not subdirectories.  Again, recall that the changes will not be in effect until you logout/login.

```
for x in `find ~/MYENV  -type f` ; do
   source $x 
done
```


## Difference between login and interactive shells
This section is based in part on on [https://stackoverflow.com/questions/18186929/what-are-the-differences-between-a-login-shell-and-interactive-shell](https://stackoverflow.com/questions/18186929/what-are-the-differences-between-a-login-shell-and-interactive-shell)

The shell that gets started when you open a window on a HPC is called a login shell.  It is also an interactive shell in that you are using it to interact with the computer.  Bash can also be run as a command.  That is, if you enter bash as a command you will start a new instance of the bash shell.  This new shell is an interactive shell but not a login shell because it was not used to do the login to the platform.   

When you start a new interactive shell the file .bashrc is sourced.  When you start a login shell the file .bash_profile is sourced. However, most versions of .bash_profile have a line that will also source .bashrc.

When you submit a slurm batch job with the command **sbatch** neither of the two files .bashrc or .bash_profile are sourced.  Note, by default, the environment you have set up at the time you run sbatch is passed to the job. 

When you start a slurm interactive session, for example using the command

```
salloc --nodes=1 --time=01:00:00 --account=$MYACCOUNT --partition=debug
```

the file .bashrc is sourced.




## Troubleshooting

The most common issue when modifying your environment is forgetting to add the previous version of PATH when you set a new one.  For example

Do this:

export PATH=/projects/myapps:$PATH

Don't do this:

export PATH=/projects/myapps

If you do the second command you will lose access to most commands and you'll need to logout/login to restore access.

Always test additions before actually implementing them.  If you use the files in ~/MYENV to modify your environment manually run the commands

```
if [ -f ~/MYENV/myvars ];    then . ~/MYENV/myvars ;    fi
if [ -f ~/MYENV/myaliases ]; then . ~/MYENV/myaliases ; fi
if [ -f ~/MYENV/mymods ];    then . ~/MYENV/mymods ;    fi
if [ -f ~/MYENV/myfuncs ];   then . ~/MYENV/myfuncs ;   fi
```

to test things.  After they are working as desired then add this lines to your .bashrc file.  You can add a # to the lines in your .bashrc file to disable them.  

There are copies of the default .bashrc and .bash_profile files in

* /etc/skel/.bash_profile
* /etc/skel/.bashrc



## Some commands



```
man — Print manual or get help for a command  EXAMPLE: man ls
man bash will show many "built in" commands in the shell

ls — List directory contents
  ls -a      Show all files, including hidden files
  ls -l      Do a detailed listing
  ls -R      Recursive listing, current directories subdirectories
  ls  *.c    List files that end in "c"
  
echo  - Prints text to the terminal window 

mkdir — Create a directory

pwd — Print working directory, that is give the name of your 
      current directory.

cd — Change directory
  cd ~  Go to your home directory
  cd .. Go up one level in the directory tree

mv — Move or rename a file or directory directory

nano - Edit a file. See above. 

rm - Remove a file
rm -r DIRECTORY will recursively remove a directory.
      Use rm -rf very carefully !DO NOT! rm -rf ~  it will wipe out 
      your home directory. 

rmdir — Remove a directory. It must be empty to be removed. It's 
        safer than rm -rf.

less — view the contents of a text file

> — redirect output from a command to a file.  Example ls > myfiles
>> - same as > except it appends to the file
> /dev/null A special case of > suppress normal output by sending 
            it the the "null file"
2> err 1> out    Send errors from a command to the file err and normal
                 output to out

1>both 2>&1 Send output and errors to the file "both"

sort - Output a sorted version of a file.  Has many options.

|  A pipe takes the standard output of one command and passes it as 
   the input to another.  Example  cat mydata | sort

cat — Read a file and send output to the terminal.  To concatenate files        
      cat one two > combined

head — Show the start of a file

tail — Show the end of a file

which - Show the location of a command.  EXAMPLE: which ls
        Which will not show bash built in commands

exit — Exit out of a shell, normally used to logout

grep - search for a string in a file(s) or output

history -  display the command history

source  -  Read  and  execute  commands  from a file

find - locate files/directories with particular characteristics.  Find 
       has many options and capabilities.  "man find"will show all the
       options.  However, an online search might be the best way to 
       deterinine the options you want.

find . -name "*xyz*"    Find all files, in the current directory and below that 
                        have names that contain xyz.
find .  -type f         Find all files, in the current directory and below.
find .  -type d         Find all directories, in the current directory and below.
find . -newermt `date +"%Y-%m-%d"` 
                        Find files that have changed today.

compgen                 Show various sets of commands
compgen -a              list all bash aliases
compgen -b              list bash builtin commands
compgen -A function     list all the bash functions.
compgen -k              list all the bash keywords
compgen -c              list all commands available to you 
compgen -c | grep file  Show commands that have "file" as part of the
                        name
```
