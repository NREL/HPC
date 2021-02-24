# Install git and bash on your laptop

In order to best enable access and use of the HPC System, you should have a terminal on your laptop that runs bash and has git available.

Contents
========

- Install `git`
- Open a bash terminal

## Install `git`

* Windows
    * Download latest verison of `git` (2.9.3 x64) from <https://git-scm.com/downloads>
    * Ensure `Git Bash Here` is selected (default)
    * Ensure `Git GUI Here` is selected (default)
    * Select `Use Git from the Windows Command Prompt` (default)
    * Select `Checkout Windows-style, commit Unix-style line endings` (default)
    * Select `Use MinTTY (the default terminal of MSYS2)` (default)
    * Hit Next for remaining
* OSX
    * If you have `git` on your machine, you can skip this step.
    * Download latest version of `git` (2.9.3) from <https://git-scm.com/downloads>
    * If using git for the first time from terminal, you may get a prompt to say GIT not installed do you want to download? Select yes (you don’t need the full Xcode)

## Open a bash terminal

* Windows
   * Open `Git Bash` from available windows applications and type the following to test.

```
cd C:
pwd
```
* Mac
   * Open `Terminal`, found in Applications->Utilities->Terminal and type the following to test
```
$ git --version
git version 2.21.1 (Apple Git-122.3)
```
