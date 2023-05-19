title: Understanding File Permissions
---

# Linux File Permissions and Ownership

Linux uses standard POSIX-style permissions to control who has the ability to read, write, or execute a file or a directory.

## Permission Levels and Ownership
Under this standard, all files and all directories have three types of permission that can be granted. 

The three permission types are:

* **r** (Read): permission to read or copy the contents of the file, but not make changes
* **w** (Write): permission to make changes, move, or delete a file
* **x** (eXecute): permission to run a file that is an executable program, such as a compiled binary, shell script, python code, etc, OR to access a directory.

Files are also "owned" by both an individual user, and a user group. This ownership is used to provide varying levels of access to different
cohorts of users on the system. 

The cohorts of users to which file permissions may be assigned include:

* **u**: permissions granted to the (u)ser who owns the file
* **g**: permissions granted to the (g)roup of users who own the file
* **o**: permissions granted to any (o)thers who are not the user or the group that own the file

On most systems, every user is a member of their own personal group that has the same name as their username, and only that user has access 
to their own personal group. Whenever a new file is created, the default is that it will be created with the user and group ownership of the 
user that created the file. However, this may not always be the case, and the permissions of the directory in which the file is created can
have an effect on the group ownership. This will be discussed in a later section.

## Viewing File and Directory Permissions

The `ls -l` command will show the ownership and permissions of a file, a list of files, or all files in a directory. Here's an example output with two files, two directories, and a symbolic link to another directory. The user in the example is "alice". 
```
[alice@el1 ~]$ ls -l 
-rwxrwx---.  1 alice alice         49 Oct 13  2020 testjob.sh
-rw-rw----.  1 alice alice         35 Mar  9 16:45 test.txt
drwxrwx---.  3 alice alice       1536 Mar 31  2021 myscripts
drwxrws---.  3 alice csc000         4096 Dec 14  2020 shared-data
lrwxrwxrwx.  1 alice alice         16 Jan 30  2023 my_proj -> /projects/csc000
```

The first field of `ls -l` output for each file consists of ten characters. These represent the permission bits for the file.

The first bit is reserved to describe the type of file. The three most common file types are:

* **-** : a dash indicates a regular file (no special file type)
* **d** : a `d` indicates that this is a directory (a type of "file" that stores a list of other files)
* **l** : an `l` indicates a symbolic link to another file/directory

The next nine bits describe the file permissions that are set. These are always in the order of read, write, and execute. 

A letter indicates that this permission is granted, a `-` indicates that the permission is not granted. 

This "**rwx**" order repeats three times: the first triplet is for User permissions, the second triplet is for Group permissions, and the third triplet is for Other permissions.

In the example above, `testjob.sh` has the permissions `-rwxrwx---`. This means that the User and Group owners have read, write, and execute permission. The last three characters are `-`, which indicates that "Other" users do not have permissions to this file.

There also may be a dot (`.`) or other character at the end of the permissions list, depending on the variety of Linux that is installed. The dot indicates that no further access controls are in place. A `+` indicates that ACLs (Access Control Lists) are in place that provide additional permissions. ACLs are an extension of the file permission system that is present on some, but not all, NREL HPC systems, and may be used to provide more fine-grained access control on a per-user basis. If the system you are using supports ACLs, you may see `man getfacl` and `man setfacl` for more help on ACLs. 

After the permissions flags is a number indicating the number of hard links to the file. It has no bearing on permissions and can be ignored.

The next two fields are the User and Group with access rights to the file. A file may only be owned by one User and one Group at a time.

### Special Permissions Flags: Setuid, Setgid, and Sticky Bits

an `s` in the e(x)ecute bit field has a special meaning, depending on whether it's in the User or Group permissions. A `t` in the "Others" 
e(x)ecute also has a special meaning.

In the Group permission bits, an `s` for the eXecute bit indicates that `SETGID` is enabled. This can be set for an individual file or for a directory, but
is most common on a directory. When setgid is enabled on a directory, any files created in the directory will have a group ownership that corresponds to the
group ownership of the directory itself, instead of the default group of the user who created the file. This is very useful when an entire directory is
intended to be used for collaboration between members of a group, when combined with appropriate group read, write, and/or execute bits.

In the User permission bits, an `s` for the eXecute bit indicates that `SETUID` is enabled. This is only used for executable files, and means that
regardless of the user who runs the program, the owner of the _process_ that starts up will be changed to the owner of the _file_. This is very
rarely used by regular users and can pose a _considerable_ security risk, because a process that belongs to a user also has access to that user's 
files as though it had been run by that user. Setuid should almost never be used.

In the Other permission bits, a `t` for the eXecute bit indicates that a "sticky bit" has been set. This is only used on directories. With the sticky bit
set, files in that directory may only be deleted by the owner of the file or the owner of the directory. This is commonly used for directories that 
are globally writeable, such as /tmp or /tmp/scratch and will be set by a system administrator. It is otherwise rarely used by regular users.

## Changing Permissions and Ownership

Only the User that owns a file may change ownership or permissions.

The `chgrp` command is used to change the Group ownership of a file or directory. 

The `chmod` command is used to change the permissions of a file or directory.

The `chown` command is used to change the User owner and/or Group owner of a file, but only system administrators may change the User owner, so this command will not be covered in this document. Please see `man chown` for more information.

### The chgrp Command

The `chgrp` command is used to change the group ownership of a file. You must be a member of the group the file currently belongs to, as well as a 
member of the destination group. 

`chgrp -c group filename`

The -c flag is recommended, as it explicitly shows any changes that are made to ownership. 

Filename can be a file, a list of files, a wildcard (e.g. `*.txt`), or a directory.

Please see `man chgrp` for more detailed information on this command.

### The chmod Command and Symbolic Permissions

The chmod command is used to change the permissions (also called file mode bits) of a file or directory. Using an alphabetic shorthand ("symbolic mode"), permissions can be changed for a file or directory, in the general format:

`chmod -c ugoa+-rwxst file`

The cohort to which permissions should be applied is first: (u)ser, (g)roup, (o)ther, or (a)ll.

The `+` or `-` following the cohort denotes whether the permissions should be added or removed, respectively.

After the +/- is the list of permissions to change: (r)ead, (w)rite, e(x)ecute are the primary attributes. (s)etuid or (s)etgid depend on the cohort
chosen: u+s is for setuid, g+s is for setgid. The s(t)icky bit may also be set.

To add eXecute permission for the User owner of a file:

`chmod u+x myscript.sh`

To add group read, write, and execute, and REMOVE read, write, execute from others:

`chmod g+rwx mydirectory`

To remove write and execute from other users:

`chmod o-wx myscript.sh`

You can also combine arguments, for example:

`chmod g+rwx,o-rwx myscript.sh`

`chmod ug+rwx,o+r,o-w myscript.sh`

Please avoid setting global read, write, and execute permissions, as it is a security risk:

`chmod a+rwx myscript.sh`

#### Using Octal Permissions With chmod

Chmod can also accept numeric arguments for permissions, instead of the symbolic permissions. This is called 
"octal" mode, as it uses base 8 (numbers 0 through 7) for binary encoding. Symbolic permissions are now generally preferred for clarity, but octal
is sometimes used as a shorthand way of accomplishing the same thing. 

In octal mode, a three or sometimes four digit number is used to represent the permission bits. The octal equivalent to "ug+rwx" is:

`chmod 770 myscript.sh`

The first position is User, the second is Group, and the last is Other.

The following table describes the value of the bit and the corresponding permission.

| bit   | permission |
|-------|------------|
| 0     | none       |
| 1     | execute    | 
| 2     | write      |
| 4     | read       |

The permission is set by the sum of the bits, from 0 to 7, with 0 being "no permissions" and 7 being "read, write, and execute."

760 and 770 are the most common for data shared by a group of users. 700 is common for protected files that should only be viewed or edited by the User who owns the file.

Occasionally there may be a fourth leading digit. This is used for setuid, setgid, or a sticky bit setting. 

#### Caution with Mode 777

The command `chmod 777` is the equivalent of `chmod a+rwx`, which grants read, write, and execute permission to ALL users on the system for the file(s) specified. Use of command should be EXTREMELY rare, and any suggestions that it be applied should be examined closely, as it poses a major security risk to your files and data. Use your best judgement.

## Further Reading About File Permissions

All of the command listed have manual pages available at the command line. See `man <command>` for more information, or `man man` for help with the manual page system itself.

Further documentation regarding file permissions and other Linux fundamentals is widely available online in text or video format, and many paper books are available. 

We do not endorse any particular source, site, or vendor. The following links may be helpful:

* https://www.redhat.com/sysadmin/linux-file-permissions-explained
* https://www.linuxfoundation.org/blog/blog/classic-sysadmin-understanding-linux-file-permissions
* https://docs.nersc.gov/filesystems/unix-file-permissions/
* https://en.wikipedia.org/wiki/File-system_permissions
* https://www.linux.com/training-tutorials/file-types-linuxunix-explained-detail/
* https://en.wikipedia.org/wiki/Unix_file_types

## Default Permissions on NREL Systems

When first created, all /projects directories will be owned by the allocation's HPC Lead User and the project's shared Group. The default permissions will typically be ug+rwx (chmod 770) or ug+rwx,o+rx (chmod 776), depending on the system. The setgid bit will also be set on the directory, so that all files created in the /projects directory will have a Group ownership of the project's group. 

## NREL Technical Help with File Permissions

The NREL HPC Support Team relies on allocation owners and users to be responsible for file permissions and ownership as a part of managing the allocation and its data, but the PI or HPC Leads of a project may request assistance in changing permissions or ownership of files that belong to the allocation by opening a support ticket with [hpc-help@nrel.gov](mailto://hpc-help@nrel.gov).

