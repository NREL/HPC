---
layout: default
title: Git
has_children: true
---

# Using Git Revision Control 

*Learn how to set up and use the Git software tool for development on the HPC systems*

Git is used locally to track incremental development and modifications to a collection of files. [GitHub](https://github.com) is a git-repository hosting web-service, which serves as a synchronized, common access point for the file collections. GitHub also has social aspects, like tracking who changed what and why. There are other git hosting services like [GitLab](https://gitlab.com) which are similar to GitHub but offer slightly different features.


NREL has a Github Enterprise server (github.nrel.gov) for internally-managed repos. Please note that github.nrel.gov is only available interally using the NREL network or VPN. NREL's git server uses SAML/SSO for logging into GitHub Enterprise. To get help accessing the server or creating a repository, please contact NREL ITS.

## Git Configuration Set Up

The git software tool is already installed on the HPC systems. 

Git needs to know your user name and an email address at a minimum:

```
$ git config --global user.name "Your name"
$ git config --global user.email "your.name@nrel.gov"
```

Github does not accept account passwords when authenticating Git operations on gitHub.com. Instead, token-based authentication ([PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) or [SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh)) is required.

### Set Up SSH authorization
The HPC systems already have SSH keys created. To set up Github SSH authorization, you can add the existing SSH (secure shell) key(s) to your Github profile. You will also need to change any remote repo URL to use SSH instead of HTTPS. 

???+ note "Set up SSH Key"
    1. On the HPC sysstem, copy the content of ~/.ssh/id_rsa.pub. 
    1. On Github, click on: your git profile >  Settings > SSH and GPG keys > New SSH key
    1. Paste the content of ~/.ssh/id_rsa.pub into the "Key" window
    1. In your local git repo directory, type:
    ```
    git remote set-url origin <git@github.nrel.gov:username/my-projectname.git>.
    ```
    Your URL can be retrived in the Github UI by going to the remote repo, then "Code" > "SSH".

!!! warning 
    Please do not alter or delete the key pair that exists on the HPC systems in ~/.ssh/. You can copy the public key to Github. 
## Git Vocabulary

??? note "Repository/repo"
    A git repository is an independent grouping of files to be tracked. A git repo has a "root" which is the directory that it sits in, and tracks further directory nesting from that. A single repo is often thought of as a complete project or application, though it's not uncommon to nest modules of an application as child repositories to isolate the development history of those submodules.

??? note "Commit"
    A commit, or "revision", is an individual change to a file (or set of files). It's like when you save a file, except with Git, every time you save it creates a unique ID (a.k.a. the "SHA" or "hash") that allows you to keep record of what changes were made when and by who. Commits usually contain a commit message which is a brief description of what changes were made.

??? note "Fork" 
    A fork is a personal copy of another user's repository that lives on your account. Forks allow you to freely make changes to a project without affecting the original. Forks remain attached to the original, allowing you to submit a pull request to the original's author to update with your changes. You can also keep your fork up to date by pulling in updates from the original.

??? note "Pull"
    Pull refers to when you are fetching in changes and merging them. For instance, if someone has edited the remote file you're both working on, you'll want to pull in those changes to your local copy so that it's up to date.
    
??? note "Pull Request"
    Pull requests are proposed changes to a repository submitted by a user and accepted or rejected by a repository's collaborators. Like issues, pull requests each have their own discussion forum. 

??? note "Push"
    Pushing refers to sending your committed changes to a remote repository, such as a repository hosted on GitHub. For instance, if you change something locally, you'd want to then push those changes so that others may access them.

??? note "Branch"
    A branch is a new/separate version of the main repository. Use branches when you want to work on a new feature, but don't want to mess-up the master while testing your ideas. 
## Tool Use

??? note "Clone an existing repo"
    For example, you could create a local working copy of the "test_repo" repo (puts it in a folder in your current directory):
    cd /some/project/dir
    git clone <git@github.nrel.gov:username/test_repo.git>
    Now, make changes to whatever you need to work on.
    Recommendation: commit your changes often, e.g., whenever you have a workable chunk of work completed.

??? note "See what files you've changed"
    `git status`

??? note "Push your changes to the repo"
    ```
    git add <filename(s)-you-changed>
    git commit -m "A comment about the changes you just made."
    git push
    ```
??? note "Get remote changes from the repo"
    If you collaborate with others in this repo, you'll want to pull their changes into your copy of the repo. You may want to do this first-thing when you sit down to work on something to minimize the number of merges you'll need to handle:
    `git pull`

??? note "Create a new local git code repo"
    # ...to be checked into your repo later:
    mkdir my.projectname
    cd my.projectname
    git init
    touch README.txt
    git add README.txt
    git commit -m 'first commit'
    git remote add origin git@hpc/my.projectname.git
    git push origin master

??? note "To push existing code to the remote repo"
    cd my.projectname
    git remote add origin git@github.nrel.gov:hpc/my.projectname.git
    git push origin master
    # you may want to stick this in your local repos .git/config:
    [branch "master"]
    remote - origin
    merge - refs/heads/master

??? note "Revert to a previous commit"
    Find the commit you want to reset to:
    git log
    Once you have the hash:
    git reset --hard <hash>
    And to push onto the remote:
    git push -f <remote> <branch>

??? note "Make a branch"
    Create a local branch called "experimental" based on the current main branch:
    ```
    git checkout main #Switch to the main branch
    git branch experimental
    ```

    Use Your Branch
    (start working on that experimental branch....):
    ```
    git checkout experimental
    git pull origin experimental
    # work, work, work, commit....:
    ```

    Send Local Branch to the Repo:
    `git push origin experimental`

    Get the Remote Repo and the Branches:
    `git fetch origin`

    Switch to the Experimental Branch:
    git checkout --track origin/experimental

    Merge the Branch into the main branch
    ```
    git checkout main
    git merge experimental
    ```
    If there are conflicts, git adds >>>> and <<<<< markers in files to mark where you need to fix/merge your code. Examine your code with git diff:
    git diff
    Make any updates needed, then git add and git commit your changes. Admittedly, there's a lot more to merging than this. Contact HPC Support if you need more details about merging and handling merge conflicts.

??? note "Delete a branch"
    Once you've merged a branch and you are done with it, you can delete it:
    `git branch --delete <branchName>` # deletes branchName from your local repo
    `git push origin --delete <branchName>` # deletes the remote branch if you pushed it to the remote server

??? note "Git diff tricks"
    You can use git log to see when the commits happened, and then git diff has some options that can help identify changes.
    What changed between two commits (hopefully back to back commits):
    ``git diff 57357fd9..4f890708 > my.patch``
    Just the files that changed:
    ``git diff --name-only 57357fd9 4f890708``

??? note "Tags"
    You can tag a set of code in git, and use a specific tagged version.
    List Tags:
    `git tags -l`
    Set a Tag:
    `git tag -a "2.2" -m"Tagging current rev at 2.2"`
    Push Your Tag
    There may be an extra step you'll want to take to ensure your tags make it up to the git server:
    `git push --tags`
    Use Tag tagname
    `git checkout tagname`

??? "Unmodify a modified file"
    What if you realize that you don't want to keep your changes to the important_code.py file? How can you easily un-modify it — revert it back to what it looked like when you last committed (or initially cloned, or however you got it into your working directory)? Luckily, git status tells you how to do that, too. In the last example output, the unstaged area looks like this:
    # Changed but not updated:
    # (use "git add <file>..." to update what will be committed)
    # (use "git checkout -- <file>..." to discard changes in working directory)
    #
    # modified: important_code.py
    It tells you pretty explicitly how to discard the changes you've made (at least, the newer versions of Git, 1.6.1 and later, do this — if you have an older version, we highly recommend upgrading it to get some of these nicer usability features). Let's do what it says:
    $ git checkout -- important_code.py
    $ git status
    # On branch master
    # Changes to be committed:
    # (use "git reset HEAD <file>..." to unstage) #
    # modified: README.txt #
    You can see that the changes have been reverted. You should also realize that this is a dangerous command: any changes you made to that file are gone — you just copied another file over it. Don't ever use this command unless you absolutely know that you don't want the file.

??? note "Point your repo to a different remote server"
    Let's say you were working on some code that you'd checked-out from a repo on some other server, say from github.com. Now, you want to check that code into NREL's repository. Once you've requested a new NREL git repo from ITS and it's configured, you can:
    `git remote set-url origin git@github.nrel.gov:hpc/my.<strongnewprojectname>.git`
    See `git help remote` for more details or you can just edit `.git/config` and change the URLs there. You're not in any danger of losing history unless you do something very silly, If you're worried, just make a copy of your repo, since your repo is your history.

??? note "Send someone a copy of your current code (not the whole repo)"
    You can export a copy of your code to your $HOME directory using the following command:
    `git archive master --prefix=my.projectname/ --output=~/my.projectname.tgz`






