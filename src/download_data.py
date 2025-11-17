"""
Script to download the Brain Tumor MRI Dataset from Kaggle.

Before running this script, ensure you have:
1. A Kaggle account
2. Kaggle API token (kaggle.json) placed in ~/.kaggle/
3. Accepted the dataset terms on Kaggle website

Usage:
    python src/download_data.py
"""

import os
import zipfile
import shutil
from pathlib import Path


def download_dataset():
    """Download the Brain Tumor MRI Dataset from Kaggle."""
    
    # Check if kaggle.json exists
    kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json_path.exists():
        print("ERROR: Kaggle API token not found!")
        print("Please download kaggle.json from https://www.kaggle.com/settings")
        print(f"and place it in {kaggle_json_path.parent}")
        return False
    
    try:
        # Import kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Authenticate
        api = KaggleApi()
        api.authenticate()
        
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        print("Downloading Brain Tumor MRI Dataset from Kaggle...")
        print("This may take a few minutes depending on your internet connection.")
        
        # Download dataset
        api.dataset_download_files(
            'masumhasan/brain-tumor-mri-dataset',
            path=data_dir,
            unzip=True
        )
        
        print(f"\nDataset downloaded successfully to {data_dir}")
        
        # List the downloaded files
        print("\nDownloaded files:")
        for item in sorted(data_dir.rglob('*')):
            if item.is_file():
                print(f"  {item}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure:")
        print("1. You have accepted the dataset terms at:")
        print("   https://www.kaggle.com/datasets/masumhasan/brain-tumor-mri-dataset")
        print("2. Your Kaggle API token is properly configured")
        return False


if __name__ == '__main__':
    success = download_dataset()
    if success:
        print("\n✓ Data download completed successfully!")
    else:
        print("\n✗ Data download failed. Please check the error messages above.")
