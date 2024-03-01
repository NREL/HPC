# Connecting With VS Code 

Microsoft Visual Studio Code (VSCode) is a popular tool for development in many programming languages, and may be used on HPC systems. However, there are some caveats to be aware of when running it remotely.

## Connecting with VSCode

To connect to an HPC system with VSCode, install the "[Remote - SSH](https://code.visualstudio.com/docs/remote/ssh)" extension from the Extensions menu. 

Press "F1" to open the command bar, and type or search for `Remote-SSH: Connect to Host...`

You may then enter your HPC username and the address of an HPC system to connect to. 

* To connect to Kestrel from the NREL VPN, enter `username@kestrel.hpc.nrel.gov`, replacing "username" with your HPC user name.

* To connect to Kestrel as an external collaborator, enter `username@kestrel.nrel.gov`, replacing "username" with your HPC user name.

Enter your HPC password (or password and OTP code if external) and you will be connected to a login node. You may open a folder on the remote host to browse your home directory and select files to edit, and so on.

## Caution About VS Code Processes

Please be aware that the Remote SSH extension runs processes on the remote host. This includes any extensions or helpers, include language parsers, code analyzers, AI code assistants, and so on. These extensions can take up a _considerable_ amount of CPU and RAM on any remote host that VS Code connects to. Jupyter notebooks loaded through VS Code will also be executed on the remote host and can use excessive CPU and RAM, as well. When the remote host is a shared login node on an HPC system, this can be a considerable drain on the resources of the login node, and cause system slowdowns for all users of that login node. 

This problem can be circumvented by using a compute node to run VS Code. This will cost AU, but will allow for full resource usage of CPU and/or RAM. 

### Kestrel

Using VSCode on a compute node will require adding an ssh key.

#### SSH Key Setup

You may use an existing key pair on your local computer/laptop, or create one with `ssh-keygen` (adding `-t ed25519` is optional, but recommended.) 

We recommend choosing a strong passphrase and storing it in a password manager. The passphrase on your key will allow you to log in via ssh, but it is not the same as your HPC account password.

!!! note "SSH Key Pair Caution"
    Do **not** replace the key pair in your Kestrel or Eagle home directory. These keys are generated when you log into the cluster, and are used by Slurm jobs to communicate between nodes. There is a corresponding public key entry in your cluster home directory ~/.ssh/authorized_keys that must also be left in place.

!!! note "Reminder About Passwords"
    Using an SSH key with an SSH agent can remove the need to use a password to SSH to Kestrel or Eagle. However, not all HPC services (including [Lex](https://hpcprojects.nrel.gov)) use SSH keys. **An SSH key does NOT replace your HPC account password**. You **must** maintain a regular HPC account password in accordance with our [Appropriate Use Policy](https://www.nrel.gov/hpc/appropriate-use-policy.html) and [User Account Password Guidelines](https://www.nrel.gov/hpc/user-account-passwords.html). Ignoring password expiration date notices will lead to automatic account lockouts, and you will need to contact [HPC Support](/Documentation/help) to restore your account.

Once you have a key pair on your local computer, use the `ssh-copy-id <username>@kestrel.hpc.nrel.gov` command to copy the public portion to Kestrel. This will add your public key to the ~/.ssh/authorized_keys file in your Kestrel home directory. Alternatively, you may manually add the contents of your PUBLIC key file (for example, the contents of ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub) onto the end of this file. **Do not delete the existing entries in these files on Kestrel or on Eagle.**

#### Editing the VSCode SSH Config File

We will now create a host entry in your local ssh config file to make connecting to Kestrel compute nodes easier. 

Use the remote-ssh command to edit your VSCode ssh config file (~/.ssh/config). Add the following:

```
Host x?00?c*
    ProxyJump <username>@kestrel.hpc.nrel.gov
```

This create a "wildcard" entry that should match Kestrel compute node names. Any time an ssh command is issued on your computer that matches the wildcard, the ssh connection will "jump" through a Kestrel login node and directly to the compute node.

If your allocation is finished on Kestrel (e.g. at the end of the FY and your allocation will not be continuing to the next) or you otherwise anticipate no further need to use VS Code with Kestrel in this fashion, you may delete this entry from your SSH config file.

#### Start a Job and Connect VSCode

SSH to Kestrel as usual (outside of VSCode) and use [sbatch](/Documentation/Slurm/batch_jobs/) or [salloc](/Documentation/Slurm/interactive_jobs) to start a job. (An interactive job with `salloc` is suggested, using a `--time` limited to only the expected duration of your working session with VS Code.)

Wait until the job has started running, and take note of the node assigned to the job. Put the terminal aside, but leave the job running.

Now use the Remote-SSH extension in VSCode to `Connect to Host...` and use the hostname of the node that your job was assigned. For example, `<username>@x1000c0s0b0n1`. 

This should open a new VSCode window that will connect to the compute node automatically. You may begin browsing your home directory and editing files in the VSCode window.

### VS Code on Eagle

Similar instructions will work for Eagle, but only for internal (NREL) users. External (non-NREL) users will not be able to use VS Code in this way. 

To use VS Code on Eagle, NREL users may follow the SSH key steps as for Kestrel, but add the following Host entry to the SSH config file, either instead of or in addition to the entry for Kestrel:

```
Host r*i*n*
    ProxyJump <username>@eagle.hpc.nrel.gov
```
