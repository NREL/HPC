# Using VirtualGL and FastX 

*VirtualGL and FastX provide remote desktop and visualization capabilities for graphical applications.*

## Remote Visualization
In addition to four standard ssh-only login nodes, Eagle is also equipped with several specialized Data Analysis and Visualization (DAV) login nodes, intended for HPC applications on Eagle that require a graphical user interface. It is not a general-purpose remote desktop, so we ask that you restrict your usage to only HPC or visualization software that requires Eagle.

There are five internal DAV nodes available only to internal NREL users (or via the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html)), and one node that is externally accessible.

All DAV nodes have 36 CPU cores (Intel Xeon Gold 6150), 768GB RAM, one 32GB NVIDIA Quadro GV100 GPU, and offer a Linux desktop (via FastX) with visualization capabilities, optional VirtualGL, and standard Linux terminal applications.

DAV nodes are shared resources that support multiple simultaneous users. CPU and RAM usage is monitored by automated software, and high usage may result in temporary throttling by Arbiter. Users who exceed 8 CPUs and 128GB RAM will receive an email notice when limits have been exceeded, and another when usage returns to normal and restrictions are removed. Please use the regular Eagle batch queue to run compute-intensive jobs in batch mode, rather than in an interactive session.

## VirtualGL
VirtualGL is an open-source package that gives any Linux remote display software the ability to run OpenGL applications with full 3D hardware acceleration. The traditional method of displaying graphics applications to a remote X server (indirect rendering) supports 3D hardware acceleration, but this approach causes all of the OpenGL commands and 3D data to be sent over the network to be rendered on the client machine. With VirtualGL, the OpenGL commands and 3D data are redirected to a 3D graphics accelerator on the application server, and only the rendered 3D images are sent to the client machine. VirtualGL "virtualizes" 3D graphics hardware, allowing users to access and share large-memory visualization nodes with high-end graphics processing units (GPUs) from their energy-efficient desktops. 

## FastX
FastX provides a means for sharing a graphical desktop. By connecting to a FastX session on a DAV node, users can run graphical applications with a similar experience to running on their workstation.  Another benefit is that you can disconnect from a FastX connection, go to another location and [reconnect to that same session](#reattaching-fastx-sessions), picking up where you left off.

## Connecting to DAV Nodes Using FastX
NREL users may use the web browser or the FastX desktop client. External users must use the FastX desktop client, or connect to the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) for the web client.

??? abstract "NREL On-Site and VPN Users" 
    ### Using Web Browser

    Launch a web browser on your local machine and connect to [https://eagle-dav.hpc.nrel.gov](https://eagle-dav.hpc.nrel.gov). After logging in with your HPC username/password you will be able to launch a FastX session by choosing a desktop environment of your choice.

    **Known Bug:**

    When launching a new session, the new session browser tab may load an error page.

    Cause: FastX Load Balancer. We see this when the load balancer redirects to the least utilized node.

    Workaround: Simply reload the page in the new session browser tab or close the tab and relaunch the active session



    **Using Desktop Client**

    Download the [Desktop Client](#download-fastx-desktop-client) and install it on your local machine, then follow these instructions to connect to one of the DAV nodes.



    **Step 1**:

    Launch the FastX Desktop Client.



    **Step 2**:

    Add a profile using the + button on the right end corner of the tool using the Web protocol.
    ![image](/assets/images/FastX/fastx-installer-image-1.png)

    **Step 3**:

    Give your profile a name and enter the settings...

    URL: <https://eagle-dav.hpc.nrel.gov>

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
    ![](/assets/images/FastX/xfce-interface-cleaned-step5.png)



??? abstract "Off-Site or Remote Users"
    Remote users must use the Desktop Client via SSH for access. NREL Multifactor token (OTP) required.

    Download the [Desktop Client](#download-fastx-desktop-client) and install it on your local machine, then follow these instructions to connect to one of the DAV nodes.



    **Step 1**:

    Launch the FastX Desktop Client.

    **Step 2**:

    Add a profile using the + button on the right end corner of the tool using the SSH protocol.
    ![Alt text](/assets/images/FastX/fastx-installer-image-1.png)

    **Step 3**:

    Give your profile a name and enter the settings...

    Host: eagle-dav.nrel.gov

    Port: 22

    Username: <HPC Username>

    ...and then save the profile.

    ![](/assets/images/FastX/eagle-dav-ssh-login-fastx-cleaned-step3.png)

    **Step 4**:

    Once your profile is saved. You will be prompted for your password+OTP_token (your multifactor authentication code) to connect.

    ![](/assets/images/FastX/eagle-dav-step4-offsite.png)

    **Step 5**:

    Select a Desktop environment of your choice and click OK.

    ![](/assets/images/FastX/eagle-dav-replacement-mate-interface-step5-offsite.png)

## Launching OpenGL Applications
You can now run applications in the remote desktop. You can run X applications normally; however, to run hardware-accelerated OpenGL applications, you must run the application prefaced by the vglrun command. 
```
$ module load matlab
$ vglrun matlab
```

## Download FastX Desktop Client

|Operating System |	Installer|
|-----------------|----------|
|Mac	          |[Download](https://starnet.com/files/private/FastX31/FastX3-3.1.22.dmg) |
|Linux	          |[Download](https://starnet.com/files/private/FastX31/FastX3-3.1.21.rhel7.x86_64.tar.gz) |
|Windows          |[Download](https://starnet.com/files/private/FastX31/FastX-3.1.22-setup.exe) |


## Multiple FastX Sessions
FastX sessions may be closed without terminating the session and resumed at a later time. However, since there is a 
license-based limit to the number of concurrent users, please fully log out/terminate your remote desktop session when
you are done working and no longer need to leave processes running. Avoid having remote desktop sessions open on multiple
nodes that you are not using, or your sessions may be terminated by system administrators to make licenses available for
active users. 

## Reattaching FastX Sessions
Connections to the DAV nodes via eagle-dav.hpc.nrel.gov will connect you to a random node. To resume a session that you have
suspended, take note of the node your session is running on (ed1, ed2, ed3, ed5, or ed6) before you close the FastX client or
browser window, and you may directly access that node when you are ready to reconnect at `ed#.hpc.nrel.gov` in the FastX client
or through your web browser at `https://ed#.hpc.nrel.gov`.

## Troubleshooting

#### Could not connect to session bus: Failed to connect to socket /tmp/dbus-XXX: Connection refused
This error is usually the result of a change to the default login environment, often by an alteration to `~/.bashrc` by 
altering your $PATH, or by configuring [Conda](https://nrel.github.io/HPC/Documentation/Software_Tools/conda/) to launch into a (base) or other environment
immediately upon login. 

For changes to your `$PATH`, be sure to prepend any changes with `$PATH` so that the default system paths are included before 
any custom changes that you make. For example: `$PATH=$PATH:/home/username/bin` instead of `$PATH=/home/username/bin/:$PATH`.

For conda users, the command `conda config --set auto_activate_base false` will prevent conda from
launching into a base environment upon login.

### How to Get Help
Please contact the [HPC Helpdesk](https://www.nrel.gov/hpc/help.html) at [hpc-help@nrel.gov](mailto://hpc-help@nrel.gov) if you have any questions, technical issues,
or receive a "no free licenses" error. 