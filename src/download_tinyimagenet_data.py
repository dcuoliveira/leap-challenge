import os
import urllib.request
import tarfile

# URL for Tiny ImageNet dataset (Stanford link)
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

# Directory where the dataset will be stored
data_dir = os.path.join(os.getcwd(), "data", "inputs")
os.makedirs(data_dir, exist_ok=True)

# Path to save the zip file
zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

# Step 1: Download the dataset
print("Downloading Tiny ImageNet dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Download completed!")

# Step 2: Extract the dataset
print("Extracting dataset...")
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(path=data_dir)
print(f"Dataset extracted to: {data_dir}")

# Step 3: Verify files
print("\nExtracted files:")
for root, dirs, files in os.walk(data_dir):
    for file in files[:10]:  # Show first 10 files
        print(os.path.join(root, file))
    break

# delete the zip file after extraction
os.remove(zip_path)
print("\nZip file removed after extraction.")
