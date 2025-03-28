# Using VirtualGL and FastX 

*VirtualGL and FastX provide remote desktop and visualization capabilities for graphical applications.*

## Remote Visualization on Kestrel
In addition to standard ssh-only login nodes, Kestrel is also equipped with several specialized Data Analysis and Visualization (DAV) login nodes, intended for HPC applications on Kestrel that require a graphical user interface. 

!!! Note About Usage
    DAV FastX nodes are a limited resource and not intended as a general-purpose remote desktop. We ask that you please restrict your usage to only HPC allocation-related work and/or visualization software that requires an HPC system.

There are seven internal DAV nodes on Kestrel available only to NREL users on the NREL VPN, on campus, or via the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) that are accessible via round-robin at **kestrel-dav.hpc.nrel.gov**. The individual nodes are named kd1 through kd7.hpc.nrel.gov.

There is also one node that is ONLY accessible by external (non-NREL) users available at **kestrel-dav.nrel.gov**. This address will connect to the node kd8, and requires both password and OTP for login. 

All Kestrel DAV nodes have 104 CPU cores (2x 52-core Intel Xeon Sapphire Rapids CPUs), 256GB RAM, 2x 48GB NVIDIA A40 GPUs, and offer a Linux desktop (via FastX) with visualization capabilities, optional VirtualGL, and standard Linux terminal applications.

DAV nodes are shared resources that support multiple simultaneous users. CPU and RAM usage is monitored by automated software called Arbiter, and high usage may result in temporary throttling of processes. 


## VirtualGL
VirtualGL is an open-source package that gives any Linux remote display software the ability to run OpenGL applications with full 3D hardware acceleration. 

The traditional method of displaying graphics applications to a remote X server (indirect rendering) supports 3D hardware acceleration, but this approach causes all of the OpenGL commands and 3D data to be sent over the network to be rendered on the client machine. With VirtualGL, the OpenGL commands and 3D data are redirected to a 3D graphics accelerator on the application server, and only the rendered 3D images are sent to the client machine. VirtualGL "virtualizes" 3D graphics hardware, allowing users to access and share large-memory visualization nodes with high-end graphics processing units (GPUs) from their energy-efficient desktops. 

## FastX
FastX provides a means to use a graphical desktop remotely. By connecting to a FastX session on a DAV node, users can run graphical applications with a similar experience to running on their workstation.  Another benefit is that you can disconnect from a FastX connection, go to another location and [reconnect to that same session](#reattaching-fastx-sessions), picking up where you left off.

## Connecting to DAV Nodes Using FastX
NREL users may use the web browser or the FastX desktop client. External users must use the FastX desktop client, or connect to the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) for the web client.


??? abstract "NREL On-Site and VPN Users" 
    ### Using a Web Browser

    Launch a web browser on your local machine and connect to [https://kestrel-dav.hpc.nrel.gov](https://kestrel-dav.hpc.nrel.gov). After logging in with your HPC username/password you will be able to launch a FastX session by choosing a desktop environment of your choice. Either [GNOME](https://www.gnome.org/) or [XFCE](https://www.xfce.org/) are available for use.


    ### Using the Desktop Client 

    Download the [Desktop Client](#download-fastx-desktop-client) and install it on your local machine, then follow these instructions to connect to one of the DAV nodes.

    **Step 1**:

    Launch the FastX Desktop Client.



    **Step 2**:

    Add a profile using the + button on the right end corner of the tool using the SSH protocol.
    ![image](../../assets/images/FastX/fastx-installer-image-1.png)

    **Step 3**:

    Give your profile a name and enter the settings...

    Address/URL: *kestrel-dav.hpc.nrel.gov*

    OR you may use the address of an individual kd or ed node if you would like to resume a previous session.

    Username: <HPC Username>

    ...and then save the profile.

    **Step 4**:

    Once your profile is saved, you will be prompted for your password to connect.



    **Step 5**:

    If a previous session exists, click (double click if in "List View") on current session to reconnect.



    OR

    **Step 5a**:

    Click the PLUS (generally in the upper right corner of the session window) to add a session and continue to step 6.

    **Step 6**:

    Select a Desktop environment of your choice and click OK to launch.
    ![](../../assets/images/FastX/kestrel-dav-mate-gnome-step5.png)



??? abstract "Off-Site or Remote Users"
    Remote users must use the Desktop Client via SSH for access. NREL Multifactor token (OTP) required.

    Download the [Desktop Client](#download-fastx-desktop-client) and install it on your local machine, then follow these instructions to connect to one of the DAV nodes.



    **Step 1**:

    Launch the FastX Desktop Client.

    **Step 2**:

    Add a profile using the + button on the right end corner of the tool using the SSH protocol.
    ![Alt text](../../assets/images/FastX/fastx-installer-image-1.png)

    **Step 3**:

    Give your profile a name and enter the settings...

    Host: kestrel-dav.nrel.gov

    Port: 22

    Username: <HPC Username>

    ...and then save the profile.

    ![](../../assets/images/FastX/kestrel-dav-ssh-login-fastx-step3-external.png)

    **Step 4**:

    Once your profile is saved. You will be prompted for your password+OTP_token (your multifactor authentication code) to connect.

    ![](../../assets/images/FastX/eagle-dav-step4-offsite.png)

    **Step 5**:

    Select a Desktop environment of your choice and click OK.

    ![](../../assets/images/FastX/kestrel-dav-mate-gnome-step5.png)

## Launching OpenGL Applications
You can now run applications in the remote desktop. You can run X applications normally; however, to run hardware-accelerated OpenGL applications, you must run the application prefaced by the vglrun command. 
```
$ module load matlab
$ vglrun matlab
```

## Choosing a GPU on Kestrel
Kestrel DAV nodes have two NVIDIA A40 GPUs. Using vglrun will default to the first GPU available, which may leave one GPU overutilized while the second is underutilized. 

To run your OpenGL software with a GPU of your choosing, you may add the `-d <gpu>` flag to vglrun to pick a GPU. The first GPU is referred to as :0.0, the second as :0.1. For example, to run Matlab on the second GPU:

`vglrun -d :0.1 matlab`

to run Ansys on the first GPU:

`vglrun -d :0.0 ansys`


## Download FastX Desktop Client

|Operating System |	Installer|
|-----------------|----------|
|Mac	          |[Download](https://www.starnet.com/files/private/FastX3/FastX3-3.3.18.dmg) |
|Linux	          |[Download](https://www.starnet.com/files/private/FastX3/FastX3-3.2.32.rhel7.x86_64.tar.gz) |
|Windows          |[Download](https://www.starnet.com/files/private/FastX3/FastX-3.3.18-setup.exe) |


## Multiple FastX Sessions
FastX sessions may be closed without terminating the session and resumed at a later time. However, since there is a 
license-based limit to the number of concurrent users, please fully log out/terminate your remote desktop session when
you are done working and no longer need to leave processes running. Avoid having remote desktop sessions open on multiple
nodes that you are not using, or your sessions may be terminated by system administrators to make licenses available for
active users. 

## Reattaching FastX Sessions
Connections to the DAV nodes via kestrel-dav.hpc.nrel.gov will connect you to a random node. To resume a session that you have suspended, take note of the node your session is running on (kd1, kd2, kd3, kd4, kd5, kd6, or kd7) before you close the FastX client or browser window, and you may directly access that node when you are ready to reconnect at e.g. `kd#.hpc.nrel.gov` in the FastX client or through your web browser at `https://kd#.hpc.nrel.gov`. 

## Troubleshooting

#### Could not connect to session bus: Failed to connect to socket /tmp/dbus-XXX: Connection refused
This error is usually the result of a change to the default login environment, often by an alteration to `~/.bashrc` by 
altering your $PATH, or by configuring [Conda](https://nrel.github.io/HPC/Documentation/Software_Tools/conda/) to launch into a (base) or other environment immediately upon login. 

For changes to your `$PATH`, be sure to prepend any changes with `$PATH` so that the default system paths are included before 
any custom changes that you make. For example: `$PATH=$PATH:/home/username/bin` instead of `$PATH=/home/username/bin/:$PATH`.

For conda users, the command `conda config --set auto_activate_base false` will prevent conda from
launching into a base environment upon login. 

#### No Free Licenses
FastX has a limited number of licenses for concurrent usage, so please remember to log out of your X session AND out of FastX when you are done working. If you receive a "no free licenses" error when trying to start a new session, please contact hpc-help@nrel.gov for assistance.

### How to Get Help
Please contact the [HPC Helpdesk](https://www.nrel.gov/hpc/help.html) at [hpc-help@nrel.gov](mailto://hpc-help@nrel.gov) if you have any questions, technical issues, or receive a "no free licenses" error. 
