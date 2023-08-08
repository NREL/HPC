--- layout: default 
title: Globus 
grand_parent: Data Movement 
parent: Transferring Data ---
# Transferring Files with Globus

*For large data transfers between NREL’s high-performance computing (HPC)
systems and another data center, or even a laptop off-site, we recommend using
Globus.*

A supporting set of instructions for requesting a Globus account and data
transfer using Globus is available on the [HPC NREL
Website](https://www.nrel.gov/hpc/globus-file-transfer.html)

## What Is Globus?

Globus provides services for research data management, including file transfer.
It enables you to **quickly, securely and reliably move your data to and from
locations you have access to**.


Globus transfers files using **GridFTP**. GridFTP is a high-performance data
transfer protocol which is optimized for high-bandwidth wide-area networks.  It
provides more reliable high performance file transfer and synchronization than
scp or rsync. It automatically tunes parameters to maximize bandwidth while
providing automatic fault recovery and notification of completion or problems.

## Globus Personal Endpoints

You can set up a "Globus Connect Personal EndPoint", which turns your personal
computer into an endpoint, by downloading and installing the Globus Connect
Personal application on your system. 

## Get a Globus Account

To get a Globus account, sign up on the [Globus account website](https://www.globusid.org/create).

### Set Up a Personal EndPoint

- Download [Globus Connect Personal](https://www.globus.org/globus-connect-personal)
- Once installed, you will be able to start the Globus Connect Personal
  application locally, and login using your previously created Globus 
  account credentials.
- Within the application, you will need to grant consent for Globus to access
  and link your identity before creating a collection that will be visible from
  the Globus Transfer website.
- Additional tutorials and information on this process is located at the Globus
  Website for both
[Mac](https://docs.globus.org/how-to/globus-connect-personal-mac/) and
[Windows](https://docs.globus.org/how-to/globus-connect-personal-windows/).

## Globus NREL Endpoints

The current NREL Globus Endpoints are:

- **nrel#eglobus** - this endpoint allows access to any files on Eagle 
(e.g., /projects, /scratch, /home, /datasets, /campaign, and /shared-projects)
- **nrel#globus-hpc1** and **nrel#globus-hpc2** - these endpoints allows access to *some* files
on Eagle (e.g., /campaign, /datasets, /shared-projects, /mss) and can be mounted
to other systems within the ESIF Data Center upon request

## Transferring Files

You can transfer files with Globus through the [Globus
Online](https://www.globus.org) website or via the [CLI](https://docs.globus.org/cli/) 
(command line interface).

??? abstract "Globus Online" 
    Globus Online is a hosted service that allows you to use a browser to transfer
    files between trusted sites called "endpoints".  To use it, the Globus software
    must be installed on the systems at both ends of the data transfer. The NREL
    endpoint is nrel#eglobus.

    1. Click Login on the [Globus web site](https://www.globus.org/). On the login
    page select "Globus ID" as the login method and click continue.  Use the Globus
    credentials you used to register your Globus.org account.  
    2. The ribbon on the left side of the screen acts as a Navigator, select File Manager
    if not already selected.  In addition, select the 'middle' option for Panels in the upper
    right, which will display space for two Globus endpoints. 
    3. The collection tab will be searchable (e.g. nrel), or **nrel#eglobus** can be 
    entered in the left collection tab.  In the box asking for authentication, **enter 
    your NREL HPC username and password**.  Do **not** use your globus.org username 
    or password when authenticating with the **nrel#eglobus** endpoint.
    4. Select another Globus endpoint, such as a personal endpoint or 
    an endpoint at another institution that you have access to.
    To use your personal endpoint, first start the Globus Connect Personal application. 
    Then search for either the endpoint name or your username in the collections tab, 
    and select your endpoint. After the first use, you should see your endpoints in 
    the recent tab when searching.  You may also setup an endpoint/directory as a bookmark.
    5. To transfer files:
        - select the files you want to transfer from one of the endpoints 
        - select the destination location in the other endpoint (a folder or directory) 
        - click the 'start' button on the source collection, and it will transfer files
          to the target collection
    6. For additional information, the [Globus Webpage](https://www.globus.org) has 
    tutorials and documentation under the Resources tab.

    *When your transfer is complete, you will be notified by email.*

??? abstract "Globus CLI (command line interface)" 
    Globus supports a command line interface (CLI), which can be used for scripting
    and automating some transfer tasks.  For more information,
    it is suggested that the user refer to the [Globus CLI](https://docs.globus.org/cli/)
    documentation located on the Globus Webpage.

    For installing **globus-cli**, the recommendation is to use a Conda environment.  In this 
    case, it is advised to follow the instructions about mixing Conda and Pip, 
    and only use Pip after establishing a base environment using Conda.  For more information about mixing Conda and Pip, refer to our internal documentation at: [Conda](../../Environment/Customization/conda.md)
    
