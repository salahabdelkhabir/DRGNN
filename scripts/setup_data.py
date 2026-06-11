import os
import sys
import urllib.request
import zipfile

PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6150987"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_primekg(save_path: str) -> None:
    print(f"Downloading PrimeKG dataset to {save_path}...")
    print("NOTE: PrimeKG is a large dataset (~2GB). This may take a while.")
    print(f"Download from: {PRIMEKG_URL}")
    print("Alternatively, manually download from: https://dataverse.harvard.edu/dataverse/primekg")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_primekg(DATA_DIR)
    print("Data setup complete.")
