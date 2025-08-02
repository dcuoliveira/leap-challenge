import os
import urllib.request
import tarfile

# URL for the CIFAR-10 Python version
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# Directory where the dataset will be stored
data_dir = os.path.join(os.getcwd(), "data", "inputs")
os.makedirs(data_dir, exist_ok=True)

# Path to save the tar.gz file
tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

# Step 1: Download the dataset
print("Downloading CIFAR-10 dataset...")
urllib.request.urlretrieve(url, tar_path)
print("Download completed!")

# Step 2: Extract the dataset
print("Extracting dataset...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=data_dir)
print(f"Dataset extracted to: {data_dir}")

# Step 3: Verify files
print("\nExtracted files:")
for root, dirs, files in os.walk(data_dir):
    for file in files:
        print(os.path.join(root, file))

# delete the tar.gz file after extraction
os.remove(tar_path)
print("\nTar file removed after extraction.")
