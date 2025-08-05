import os
import urllib.request
import zipfile

# URL of the Tiny ImageNet 224x224 dataset
url = "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/tiny-224.zip"

# Directory to store data
data_dir = os.path.join(os.getcwd(), "data", "inputs")
os.makedirs(data_dir, exist_ok=True)

# Path to download the zip file
zip_path = os.path.join(data_dir, "tiny-224.zip")

# Step 1: Download dataset
print("Downloading Tiny ImageNet 224 dataset...")
urllib.request.urlretrieve(url, zip_path)
print(f"Download completed! File saved to: {zip_path}")

# Step 2: Extract dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print(f"Dataset extracted to: {data_dir}")

# Step 3: Remove zip file to save space
os.remove(zip_path)
print("Zip file removed after extraction.")

# Step 4: Verify extracted files
print("\nExtracted folder structure:")
for root, dirs, files in os.walk(data_dir):
    level = root.replace(data_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # print first 5 files per folder
        print(f"{subindent}{file}")
