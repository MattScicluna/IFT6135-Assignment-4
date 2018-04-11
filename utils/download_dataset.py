import urllib.request
import shutil
import zipfile

URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1'
file_name = 'celebA.zip'

print("Downloading...")
# Download the file from `url` and save it locally under `file_name`:

with urllib.request.urlopen(URL) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

print("Extracting file...")
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall('../data')
zip_ref.close()
