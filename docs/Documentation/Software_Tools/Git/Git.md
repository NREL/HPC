---
layout: default
title: Git
has_children: true
---

# Git

To begin, let's start by clearing up some common misconceptions about git. You probably have heard of git through the popular git-repository hosting web-service [GitHub](https://github.com). GitHub (the hosting service) is to git (the open-source version control software) what the internet is to computers&mdash;git is used locally to track incremental development and modifications to a collection of files, and GitHub gets those changes to serve as a synchronized, common access point. GitHub also has social aspects, like tracking who changed what and why. There are other git hosting services like [GitLab](https://gitlab.com) which are similar to GitHub but offer slightly different features.

The git workflow has some pretty colorful vocabulary, so let's define some of the terms to avoid confusion going forward:
* **Repository/repo**: A git repository is an independent grouping of files to be tracked. A git repo has a "root" which is the directory that it sits in, and tracks further directory nesting from that. A single repo is often thought of as a complete project or application, though it's not uncommon to nest modules of an application as child repositories to isolate the development history of those submodules.
  
* **Commit**: A commit, or "revision", is an individual change to a file (or set of files). It's like when you save a file, except with Git, every time you save it creates a unique ID (a.k.a. the "SHA" or "hash") that allows you to keep record of what changes were made when and by who. Commits usually contain a commit message which is a brief description of what changes were made.

* **Fork**: A fork is a personal copy of another user's repository that lives on your account. Forks allow you to freely make changes to a project without affecting the original. Forks remain attached to the original, allowing you to submit a pull request to the original's author to update with your changes. You can also keep your fork up to date by pulling in updates from the original.

* **Pull**: Pull refers to when you are fetching in changes and merging them. For instance, if someone has edited the remote file you're both working on, you'll want to pull in those changes to your local copy so that it's up to date.

* **Pull request:** Pull requests are proposed changes to a repository submitted by a user and accepted or rejected by a repository's collaborators. Like issues, pull requests each have their own discussion forum. 

* **Push**: Pushing refers to sending your committed changes to a remote repository, such as a repository hosted on GitHub. For instance, if you change something locally, you'd want to then push those changes so that others may access them.

