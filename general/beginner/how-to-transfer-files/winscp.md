# Transferring files using Windows

*WinSCP for can used to securely transfer files between your local computer running Microsoft Windows and a remote computer running Linux.*

### Setting Up WinSCP

- Download and install [WinSCP](https://winscp.net/eng/docs/guides#setup).

*You may follow the prompts to import your PuTTY sites to simplify host management.*

### Connecting to a Host

- Set up a host (if needed) by selecting "New Site" and providing a host name (e.g. peregrine.nrel.gov) and your user name.  In most cases, use the SFTP protocol.
- Connect to the server by selecting a site and clicking [Login].
- Enter your password or Password+OTP Token when prompted.

### Transferring Files

You may use WinSCP to transfer individual files or to synchronize the Local Directory to the Remote Directory.

Transfer files by dragging them from the Local Directory (left pane) to the Remote Directory (right pane) or vice versa.  Once the transfer is complete the selected file will be visible in the Remote Directory pane.

Synchronizing directories allows you to easily replicate changes affecting entire directory structures back and forth.  To synchronize the Remote Directory and the Local Directory select Synchronize from the Commands menu. Select the Synchronize Files mode and click OK.
