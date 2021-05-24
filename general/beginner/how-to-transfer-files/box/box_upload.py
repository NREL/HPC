import argparse
import glob

from boxsdk import Client, OAuth2

parser = argparse.ArgumentParser(description="run hive")
parser.add_argument(
    'file_path',
    help='which file(s) to upload to box?'
)
parser.add_argument(
    '--client_id',
    help='client id from the box developer app'
)
parser.add_argument(
    '--client_secret',
    help='client secret from the box developer app'
)
parser.add_argument(
    '--access_token',
    help='developer token from the box developer app'
)
parser.add_argument(
    '--folder_id',
    help='destination folder id from the box folder url (like 137327009947)'
)


def run():
    args = parser.parse_args()

    # these all come from the box developer app
    if not args.client_id:
        client_id = input("Client ID:")
    else:
        client_id = args.client_id

    if not args.client_secret:
        client_secret = input("Client Secret:")
    else:
        client_secret = args.client_secret

    if not args.access_token:
        access_token = input("Access Token:")
    else:
        access_token = args.access_token

    # pull this from the box url
    if not args.folder_id:
        folder_id = input("Folder ID:")
    else:
        folder_id = args.folder_id

    oauth2 = OAuth2(client_id, client_secret, access_token=access_token)
    client = Client(oauth2)

    # which files to upload
    files = glob.glob(args.file_path)

    for f in files:
        print(f"uploading {f}")
        response = client.folder(folder_id).upload(f)

    print(f"finished uploading {len(files)} files!")


if __name__ == "__main__":
    run()
