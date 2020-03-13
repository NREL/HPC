# Transfering Files with Globus

*For large data transfers between NREL’s high-performance computing (HPC) systems and another data center, or even a laptop off-site, we recommend using Globus.*

A supported set of instructions for requesting a HPC Globus account and data transfer using Globus is available on the [HPC NREL Website](https://www.nrel.gov/hpc/globus-file-transfer.html)

### What Is Globus?

Globus provides services for research data management, including file transfer. It enables you to **quickly, securely and reliably move your data to and from locations you have access to**.

Globus transfers files using **GridFTP**. GridFTP is a high-performance data transfer protocol which is optimized for high-bandwidth wide-area networks.  It provides more reliable high performance file transfer and synchronization than scp or rsync. It automatically tunes parameters to maximize bandwidth while providing automatic fault recovery and notification of completion or problems.

### Globus Personal endpoints

You can set up a "Globus Connect Personal EndPoint", which turns your personal computer into an endpoint, by downloading and installing the Globus Connect Personal application on your system. We use a personal endpoint to demonstrate how to transfer files to and from Peregrine.

###### Set Up a Personal EndPoint

- Login to the Globus website. From the Manage Data drop down menu, select Transfer Files. Then click Get Globus Connect Personal.
- Pick a name for your personal endpoint and select Generate Startup Key. Follow the instructions on the web page to save your key.
- Download and install the Globus Connect Personal software on your personal system.
- Copy the startup key from the Globus web page to this application.

###### Configure Permissions on a Personal EndPoint

Once Globus Connect Personal is installed on your system, set up the permissions for reading or writing files to your local system.

- If you are using a Mac, click on the "g" icon on the upper right portion of your screen to access the Globus Connect Personal application.
- Select Preferences. To allow Globus to copy files to your local system, make sure that the directory (folder) they will go in is Writable.

### Transferring Files

You can transfer files with Globus through the [Globus Online](https://www.globus.org) website or via a CLI (command line interface).

#### Globus Online

Globus Online is a hosted service that allows you to use a browser to transfer files between trusted sites called "endpoints".  To use it, the Globus software must be installed on the systems at both ends of the data transfer. The NREL endpoint is nrel#globus.

- Click Login on the [Globus web site](https://www.globus.org/). On the login page select "Globus ID" as the login method and click continue.  Use the Globus credentials you used to register your Globus.org account.
- Go to the Transfer Files page, the link is located under the Manage Data tab at the top of the page.
- Select **nrel#globus** as the endpoint on one right side. In the box asking for authentication, **enter your Peregrine (NREL HPC) username and password**. Do not use your globus.org username and password when authenticating with the nrel#globus endpoint.
- Select another Globus endpoint, such as a personal endpoint or an endpoint at another institution that you have access to. To use your personal endpoint, first start the Globus Connect Personal application. Then enter "USERNAME#ENDPOINT" on the left side or use the drop down menu to find it. Click "go".
- To transfer files
  - Select the files you want to transfer someplace else from the system from the dialog box on the left.
  - Select the destination location (a folder or directory) from the dialog box on right right.
  - Click the large blue button at the top of the screen to begin to transfer the files.

*When your transfer is complete, you will be notified by email.*

#### Globus CLI (Command line Interface)

Configuring your Globus.org account to allow ssh CLI access

To use the CLI you must have a Globus account with ssh access enabled. To enable your account for ssh access you must add your ssh public key to your Globus account by visiting the [Manage Identities page](https://www.globus.org/account/ManageIdentities) and clicking "manage SSH and X.509 keys" and then "Add a New Key". If you do not have an ssh key, follow the [directions here](https://docs.globus.org/faq/command-line-interface/#how_do_i_generate_an_ssh_key_to_use_with_the_globus_command_line_interface) to create one.

Globus.org CLI examples

```shell
$ ssh <globus username>@cli.globusonline.org <command>  <options> <params>
$ ssh <globus username>@cli.globusonline.org help
```

A one-liner can be used to integrate globus.org CLI commands into shell scripts

```shell
$ ssh <globus username>@cli.globusonline.org scp nrel#globus:/globusro/file1.txt myuser#laptop:/tmp/myfile.txt
```

The globus.org CLI can be used interactively

```shell
$ ssh <globus username>@cli.globusonline.org
Welcome to globusonline.org, <globus username>. Type 'help' for help.
$ help
$ scp nrel#globus:/globusro/file1.txt myuser#laptop:/tmp/myfile.txt
$ exit
```

*You can find more information on the Globus CLI from the [official Globus CLI documentation](https://docs.globus.org/cli/examples/).*
