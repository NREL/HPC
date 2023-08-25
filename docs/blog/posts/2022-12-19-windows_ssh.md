---
date: 2022-12-19
authors: 
  - aco
---

# Workaround for Windows SSH "Corrupted MAC on input" Error

Some people who use Windows 10/11 computers to ssh to Eagle from a Windows command prompt, powershell, or via Visual Studio Code's SSH extension might receive an error message about a "Corrupted MAC on input" or "message authentication code incorrect." This error is due to an outdated OpenSSL library included in Windows and a security-mandated change to ssh on Eagle. However, there is a functional workaround for this issue. (Note: If you are not experiencing the above error, you do not need and should not use the following workaround.)

<!-- more -->

For command-line and Powershell ssh users, adding `-m hmac-sha2-512` to your ssh command will resolve the issue. For example: `ssh -m hmac-sha2-512 <username>@eagle.hpc.nrel.gov`.

For VS Code SSH extension users, you will need to create an ssh config file on your local computer (~/.ssh/config), with a host entry for Eagle that specifies a new message authentication code: 
```
Host eagle
    HostName eagle.hpc.nrel.gov
    MACs hmac-sha2-512
```

The configuration file will also apply to command-line ssh in Windows. This [Visual Studio Blog post](https://code.visualstudio.com/blogs/2019/10/03/remote-ssh-tips-and-tricks) has further instructions on how to create the ssh configuration file for Windows and VS Code.

...
