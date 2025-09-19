# download_datasets.py
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import gdown

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)

def download_nerf_datasets():
    """Download NeRF datasets"""
    data_dir = Path("benchmark/data")
    data_dir.mkdir(exist_ok=True)
    
    # NeRF Blender synthetic dataset
    print("Downloading NeRF synthetic dataset...")
    gdown.download(
        "https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1",
        str(data_dir / "nerf_blender.zip"),
        quiet=False
    )
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(data_dir / "nerf_blender.zip", 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Download complete!")

if __name__ == "__main__":
    download_nerf_datasets()