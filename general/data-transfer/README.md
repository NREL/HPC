# Shell wrappers for common `rsync` tasks

- `rsync_upload.sh` : Upload to NREL HPC system.
- `rsync_download.sh` : Download from NREL HPC system.
- `rsync_sync.sh` : Sync data between locations.

### Example usage

```
bash rsync_upload.sh FILE_TO_UPLOAD HPC_USERNAME NREL_CLUSTER REMOTE_LOCATION
bash rsync_download.sh FILE_TO_DOWNLOAD HPC_USERNAME NREL_CLUSTER REMOTE_LOCATION
bash rsync_sync.sh FILE_TO_SYNC SYNC_TO_LOCATION
```
