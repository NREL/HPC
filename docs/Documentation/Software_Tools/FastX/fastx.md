---
layout: default
title: FastX 
parent: Environments
---
# FastX on Eagle DAV nodes
In addition to four standard ssh-only login nodes, Eagle is also equipped with several specialized Data Analysis and Visualization (DAV) login nodes, intended for HPC applications on Eagle that require a graphical user interface. It is not a general-purpose remote
desktop, so we ask that you restrict your usage to only HPC or visualization software that requires Eagle.

There are five internal DAV nodes available only to internal NREL users (or via the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html)), and one node that is externally accessible. 

All DAV nodes have 36 CPU cores (Intel Xeon Gold 6150), 768GB RAM, one 32GB NVIDIA Quadro GV100 GPU, and offer a 
Linux desktop (via FastX) with visualization capabilities, optional VirtualGL, and standard Linux terminal applications.

DAV nodes are shared resources that support multiple simultaneous users. CPU and RAM usage is monitored by automated software, and 
high usage may result in temporary throttling by Arbiter. Users who exceed 8 CPUs and 128GB RAM will receive an email 
notice when limits have been exceeded, and another when usage returns to normal and restrictions are removed.

## Getting Started with FastX

Information on how to log into a DAV node with a FastX remote desktop can be found in the [FastX Documentation](https://www.nrel.gov/hpc/eagle-software-fastx.html) at [https://www.nrel.gov/hpc/eagle-software-fastx.html](https://www.nrel.gov/hpc/eagle-software-fastx.html). NREL users may use the web browser or the FastX desktop client. External users must use the FastX desktop client, or connect to
the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) for the web client.

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
