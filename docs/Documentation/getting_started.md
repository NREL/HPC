---
title: Getting Started

[Learn how to request an NREL HPC user account.](https://www.nrel.gov/hpc/user-accounts.html)

## Frequently Asked Questions

??? note "What is high-performance computing?"

    Generally speaking, HPC infrastructure is coordinating many discrete units capable
    of independent computation to cooperate on portions of a task to complete far more
    computation in a given amount of time than any of the units could do individually.
    In other words, an HPC system is lots of individual computers working together.

??? note "Is NREL HPC related to the Information Technology Services Desk?"

    HPC Operations and Information Technology Services (ITS) are separate groups with
    different responsibilities. ITS will handle issues with your workstation or any other
    digital device you are issued by NREL. HPC Operations will assist with issues regarding
    HPC systems.  **Note that your NREL HPC account is a different account from your ITS credentials**
    that you use to login to your workstation, e-mail, and the many other IT services
    provided by the Service Desk.

??? note "What are project allocations?"

    Over the fiscal year, there is a given amount of time each computer in the HPC system(s)
    can be expected to be operational and capable of performing computation. HPC project
    allocations allocate a portion of the total assumed available computing time. The sum of all awarded project
    allocations' compute-time approximates the projected availability of the entire system.
    Project allocations are identified by a unique "handle" which doubles as a Linux account
    under which you submit HPC jobs related to the project to the job scheduler. Learn
    more about [requesting an allocation](https:/www.nrel.gov/hpc/resource-allocation-requests.html).

??? note "How can I access NREL HPC systems?"

    Begin by [requesting an NREL HPC account](https:/www.nrel.gov/hpc/user-accounts.html). 
    Then, consult our guide on [how to connect to the NREL HPC system](https:/www.nrel.gov/hpc/system-connection.html).

??? note "What is a one-time password (OTP) token?"

    OTP tokens are a means of two-factor authentication by combining a temporary (usually
    lasting 60 seconds) token to use along with your account password. Tokens are generated
    using the current time stamp and a secure hashing algorithm.  **Note that you only need an 
    OTP to access systems outside the NREL firewall**, namely if you are an external collaborator. 
    NREL employees can be on-site or use a VPN to access HPC systems via the *.hpc.nrel.gov domain.

??? note "What is a virtual private network (VPN)?"

    VPNs simulate being within a firewall (which is an aggressive filter on inbound network
    traffic) by encapsulating your traffic in a secure channel that funnels through the
    NREL network. While connected to a VPN, internal network domains such as *.hpc.nrel.gov
    can be accessed without secondary authentication (as the VPN itself counts as a secondary
    authentication). NREL employees may use the NREL VPN while external collaborators
    may use the NREL HPC VPN using their OTP token. This provides the convenience of not
    having to continually type in your current OTP token when accessing multiple systems
    in a single session.

??? note "What is a "job?""

    This is the general term used for any task submitted to the HPC systems to be queued
    and wait for available resources to be executed. Jobs vary in how computationally
    intensive they are.

??? note "What is a "node?""

    A node is a complete, independent system with its own operating system and resources,
    much like your laptop or desktop. HPC nodes are typically designed to fit snugly in
    tight volumes, but in principle you could convert several laptops into a cluster,
    and they would then be "nodes."

??? note "What are "login" and "compute" nodes?"

    Login nodes are the immediate systems your session is opened on once you successfully
    authenticate. They serve as preparation systems to stage your user&nbsp;environment and
    launch jobs. These login nodes are shared resources, and because of that the HPC team
    employs a program called Arbiter2 to ensure that these resources aren't being used
    inappropriately (see 'What is proper NREL HPC login node etiquette' for more detail).
    Compute nodes are where your jobs get *computed* when submitted to the scheduler. 
    You gain exclusive access to compute nodes that are executing your jobs, whereas there 
    are often many users logged into the login nodes at any given time.

??? note "What is proper NREL HPC login node etiquette?"

    As mentioned above, login nodes are a shared resource, and are subject to process
    limiting based on usage. Each user is permitted up to 8 cores and 100GB of RAM at
    a time, after which the Arbiter monitoring software will begin moderating resource
    consumption, restricting further processes by the user until usage is reduced to acceptable
    limits. If you do computationally intensive work on these systems, it will unfairly
    occupy resources and make the system less responsive for other users. Please reserve
    your computationally intensive tasks (especially those that will fully utilize CPU
    cores) for jobs submitted to compute nodes. Offenders of login node abuse will be
    admonished accordingly. For more information please see our [policy](https:/www.nrel.gov/hpc/inappropriate-use-policy.html) on what 
    constitutes inappropriate use.

??? note "What is "system time?""

    System time is a regularly occurring interval of time during which NREL HPC systems
    are taken offline for necessary patches, updates, software installations, and anything
    else to keep the systems useful, updated, and secure. **You will not be able to access 
    the system or submit jobs during system times.**  System times occur the first Monday 
    every month. A reminder announcement is sent out prior to every system time detailing 
    what changes will take place, and includes an estimate of how long the system time will be. 
    You can check the [system status page](https:/www.nrel.gov/hpc/system-status.html) if you are ever 
    unsure if an NREL HPC system is currently down for system time.

??? note "How can I more closely emulate a Linux/macOS workflow on my Windows workstation?"

    As you become familiar with navigating the HPC Linux systems you may come to prefer
    to use the same command-line interfaces locally on your workstation to keep your workflow
    consistent. There are many terminal emulators that can be used on Windows which provide
    the common Linux and macOS command-line interface. The official Linux command-line
    emulator for Windows is known as the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10). 
    Other recommended terminal applications include: [Git Bash](https://git-scm.com/downloads), [Git for WIndows](https://gitforwindows.org/), 
    [Cmder](https://cmder.app/), and [MYSYS2](https://www.msys2.org/). Note that PuTTY is not a terminal emulator, 
    it is only an SSH client. The applications listed above implement an <kbd>ssh</kbd> command, 
    which mirrors the functionality of PuTTY.

??? note "What is the secure shell (SSH) protocol?"

    Stated briefly, the SSH protocol establishes an encrypted channel to share various
    kinds of network traffic. Not to be confused with the <kbd>ssh</kbd> terminal command or 
    SSH clients which are applications that implement the SSH protocol in software to 
    create secure connections to remote systems.

??? note "Why aren't my jobs running?"

    Good question! There may be hundreds of reasons why. Please [contact HPC support](https:/www.nrel.gov/hpc/contact-us.html)
    with a message containing as many relevant details as you can provide so we are more
    likely to be able to offer useful guidance (such as what software you're using, how
    you are submitting your job, what sort of data you are using, how you are setting
    up your software environment, etc.).

---
