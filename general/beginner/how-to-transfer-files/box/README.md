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


Then, you can clone this repo, create an environment and install the script:

```
git clone https://github.com/NREL/HPC.git

cd HPC/general/beginner/how-to-transfer-files/box

conda create -n box python=3.8
source activate box

pip install -e . 
```


### running

Running the script can be done by modifying the command below with your own parameters: 

```
boxupload my_file.file \
--client_id ZvS6ZikzMPzJrKFriUETMeQpGMG5rZ \
--client_secret S5BJTtv9Tnz95cZSvW7amaCNtUqVFP \
--access_token tRF3JCS5SfT7xwZnHtfiY3pYM8pn48 \
--folder_id 137327009947
```

The `--folder_id` can be found by looking at the box url when you navigate to the folder you want to upload to:

```
https://nrel.app.box.com/folder/132339394972
```

The `--client_id`, `--client_secret` and `--access_token` are all found in the developer app that you created under the "Configuration" section:

Note: you can omit the arguments `--client_secret` or `--access_token` and the script will prompt you for those
without any clear text.

![Image of Box App](/assets/developer_box.png)
