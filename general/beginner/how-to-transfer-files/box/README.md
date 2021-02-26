# Transferring Files to Box   

A programatic way to transfer files from HPC to Box

## box_upload.py

A simple script for uploading files from an HPC resource to the nrel.box.com site. 

### setup

First you'll need to set up a new box developer app and get an authentication token 

    1. navigate to https://nrel.app.box.com/developers/console/
    2. select "Create New App"
    3. select "Custom App"
    4. select "Standard OAuth 2.0 (User Authentication)"
    5. name the app (something like "api_access")
    6. you should get an access token that will be active for 60 minutes; if you need a new one you can go to "Configuration" -> "Generate Developer Token"

Next you'll have to install the boxsdk package on the remote resource, something like:

```
conda create -n box python=3.8
conda activate box
pip install boxsdk
```

Then, you can clone this repo and update the script to match your situation:

```
git clone https://github.com/NREL/HPC.git
vim HPC/general/beginner/how-to-transfer-files/box/box_upload.py
```

The `FOLDER_ID` can be found by looking at the box url when you navigate to the folder you want to upload to:

```
https://nrel.app.box.com/folder/132339394972
```

### running

Running the script is as simple as `python box_upload.py`

The `CLIENT_ID`, `CLIENT_SECRET` and `ACCESS_TOKEN` are all found in the developer app that you created under the "Configuration" section:

![Image of Box App](/assets/developer_box.png)
