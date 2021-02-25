from boxsdk import Client, OAuth2

import getpass
import glob

# these all come from the box developer app
CLIENT_ID = input("Client ID:")
CLIENT_SECRET = getpass.getpass("Client Secret:")
ACCESS_TOKEN = getpass.getpass("Access Token:")

# pull this from the box url
FOLDER_ID = 132339394972

# update this to point to the files you want to upload
UPLOAD_FILE_PATTERN = "path/to/my/files/*.json"

oauth2 = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)
client = Client(oauth2)

# which files to upload
files = glob.glob(UPLOAD_FILE_PATTERN)

for f in files:
        print(f"uploading {f}")
        response = client.folder(FOLDER_ID).upload(f)
