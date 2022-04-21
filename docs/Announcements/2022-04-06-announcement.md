---
title: April 2022 Monthly Update
data: 2022-04-06
layout: default
brief: FY23 Allocations, Documentation, Eagle Login Nodes, CSC Tutorial
---

# FY23 HPC Allocation Process
The Eagle allocation process for FY23 is scheduled to open up on May 11, with applications due June 8. The application process will be an update of the process used in FY23, with additional information requested to help manage the transition from Eagle to Kestrel. HPC Operations will host a webinar on May 17 to explain the application process.  Watch for announcements.

# Documentation
We would like to announce our user-contributed [documentation repository](https://github.com/NREL/HPC) and [website](https://nrel.github.io/HPC/) for Eagle and other NREL HPC systems that is open to both NREL and non-NREL users. This repository serves as a collection of code examples, executables, and utilities to benefit the NREL HPC community. It also hosts a site that provides more verbose documentation and examples.  If you would like to contribute or recommend a topic to be covered please open an issue or a pull request in the repository. Our [contribution guidelines](https://github.com/NREL/HPC/blob/master/CONTRIBUTING.md) offer more detailed instructions on how to add content to the pages.

# Eagle login node etiquette
Eagle logins are shared resources that are heavily utilized. We have some controls in place to limit per user process use of memory and CPU that will ramp down your processes usage over time. We recommend any sustained heavy usage of memory and CPU take place on compute nodes, where these limits aren't in place. If you only need a node for an hour, nodes in the debug partition are available. We permit compiles and file operations on the logins, but discourage multi-threaded operations or long, sustained operations against the file system. We cannot put the same limits on file system operations as memory and CPU, therefore if you slow the file system on the login node, you slow it for everyone on that login. Lastly, Fastx, the remote windowing package on the ED nodes, is a licensed product. When you are done using FastX, please log all the way out to ensure licenses are available for all users.

# CSC Tutorials Team - External Users
Staff in the Computational Science Center host multiple tutorials and workshops on various computational science topics throughout the year, such as Visualization, Cloud, HPC, and others.  In Microsoft Teams, a “[Computational Sciences Tutorials](https://teams.microsoft.com/dl/launcher/launcher.html?url=%2F_%23%2Fl%2Fteam%2F19%3A6nLmPDt9QHQMEuLHVBaxfsitEZSGH6oXT6lyVauMvXY1%40thread.tacv2%2Fconversations%3FgroupId%3D22ad3c7b-a45a-4880-b8b4-b70b989f1344%26tenantId%3Da0f29d7e-28cd-4f54-8442-7885aee7c080&type=team&deeplinkId=9129ae82-7ea2-4eaa-ab17-f3bb9d75cf5c&directDl=true&msLaunch=true&enableMobilePage=true&suppressPrompt=true)” public team was just created to be the hub for all such tutorials and workshops.

As an external user, you will be able to view discussion board posts, resource files, our SharePoint Calendar, and lists of the upcoming schedule and related repo links. Unfortunately, you will not be able to access recordings or the survey.  The SharePoint Calendar provides a month view for upcoming tutorials, their descriptions, and links to join. If you miss the monthly announcements in our newsletters, you can access calendar events and find meeting information to join the tutorials in the Teams channel. The Upcoming Schedule provides a list view of the upcoming events and their tentative dates. 

For external users, you will receive an invite from the team.  Should you decide to join the public team, there a few steps you will need to take.  First, you need go through the steps to register a free Office365 account (or login if you already have an account). Next, you will need to download Microsoft Authenticator or another authenticator application.  The process is straightforward, and you will be prompted during each step of the process.  If you do not accept the invite or do not wish to go through the process of joining the public team, you can rely on the monthly newsletters or visit the [Training Page](https://www.nrel.gov/hpc/training.html) on [https://hpc.nrel.gov](https://hpc.nrel.gov) for meeting information. 

## Instructions:

* You will receive a welcome email from the team owner (sometime this week), with information about the team.  Click on accept. 
* If you have never created a MS Office 365 account, you will prompted to create one. If you already have a MS Office 365 account, login. 
* The first time you log in, you will be prompted to set up Microsoft Authenticator or other authenticator app.
* From your mobile device, Download and install the app from the Apple Store (for iOS) or the Google Play Store (for Android) and Open the app.
* On your mobile device, you will be prompted to allow notifications. Select Allow.
* On your mobile device, click OK on the screen for what information Microsoft gathers.
* Click Skip on the "Add personal account" page.
* Click Skip on the "Add non-Microsoft account" page.
* Click Add Work Account on the "Add work account" page.
* Click OK to allow access to the camera.
* Going forward, anytime you login, you will get a prompt on your phone to authenticate. 
