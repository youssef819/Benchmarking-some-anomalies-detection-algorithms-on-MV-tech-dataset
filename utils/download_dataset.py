import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import kagglehub
import zipfile
import shutil
import glob

class MVTecDownloader:
    def __init__(self, output_dir="./data/mvtec"):
        """
        Initialize MVTec dataset downloader
        
        Args:
            output_dir (str): Directory where to save the dataset
        """
        self.output_dir = Path(output_dir)
        self.dataset_id = "ipythonx/mvtec-ad"
        self.home = os.path.expanduser("~")
        self.kaggle_dir = os.path.join(self.home, ".kaggle")
        self.kaggle_json = os.path.join(self.kaggle_dir, "kaggle.json")
        self.stats = None

    def find_zip_file(self, base_path):
        """Find the first ZIP file in the download directory."""
        zip_files = glob.glob(os.path.join(base_path, "*.zip"))
        if not zip_files:
            return None
        return zip_files[0]

    def download(self):
        """Download the MVTec AD dataset and return the root path of the extracted dataset."""
        
        print(f"Checking for existing dataset in {self.output_dir}...")
    
        # Define the dataset root directory
        dataset_root = os.path.join(self.output_dir, "mvtec_anomaly_detection")
        
        # Verify if the dataset is already downloaded and extracted correctly
        if os.path.exists(dataset_root) and self._verify_dataset_structure(dataset_root):
            print("Dataset already exists and appears valid. Skipping download.")
            return dataset_root
            
        print(f"Downloading MVTec AD dataset to {self.output_dir}...")
        
        try:
            # Ensure kaggle.json exists
            if not os.path.exists(self.kaggle_json):
                raise FileNotFoundError(
                    f"Missing kaggle.json in {self.kaggle_dir}. "
                    "Please create a Kaggle account, get your API credentials from "
                    "https://www.kaggle.com/settings, and save them to kaggle.json"
                )
            
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Clean up any partial previous downloads
            if os.path.exists(dataset_root):
                shutil.rmtree(dataset_root)
            
            # Use kagglehub to download the dataset
            print("Downloading dataset...")
            base_path = kagglehub.dataset_download(
                self.dataset_id,
                force_download=True
            )
            
            # Find the ZIP file
            zip_path = self.find_zip_file(base_path)
            if not zip_path:
                raise FileNotFoundError(f"No ZIP file found in download directory {base_path}")
            
            print(f"Found downloaded archive at {zip_path}")
            print(f"Extracting dataset to {self.output_dir}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Look at the contents of the ZIP file
                contents = zip_ref.namelist()
                print(f"ZIP file contains {len(contents)} files")
                print("First few files:", contents[:5])
                
                # Extract the contents
                zip_ref.extractall(self.output_dir)
            
            # Find the actual extracted directory
            extracted_dirs = [d for d in os.listdir(self.output_dir) 
                            if os.path.isdir(os.path.join(self.output_dir, d))]
            print(f"Found directories after extraction: {extracted_dirs}")
            
            if not extracted_dirs:
                raise RuntimeError("No directories found after extraction")
            
            # Update dataset_root if necessary
            if "mvtec_anomaly_detection" not in extracted_dirs:
                actual_dir = extracted_dirs[0]
                dataset_root = os.path.join(self.output_dir, actual_dir)
                print(f"Using {actual_dir} as dataset root")
            
            # Verify the extraction worked
            if not self._verify_dataset_structure(dataset_root):
                raise RuntimeError("Dataset extraction appears incomplete or corrupted")
            
            print("Dataset downloaded and extracted successfully!")
            return dataset_root
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            if os.path.exists(dataset_root):
                shutil.rmtree(dataset_root)
            sys.exit(1)

    def _verify_dataset_structure(self, dataset_root):
        """Helper method to verify basic dataset structure"""
        if not os.path.exists(dataset_root):
            return False
            
        expected_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
        
        # Check for at least one category directory
        return any(
            os.path.exists(os.path.join(dataset_root, cat))
            for cat in expected_categories
        )

    def verify_dataset(self, dataset_root):
        """Verify the downloaded dataset structure and integrity"""
        print("Verifying dataset structure...")
        
        if not os.path.exists(dataset_root):
            print(f"Error: Dataset root directory {dataset_root} does not exist!")
            return False
            
        expected_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
        
        missing_categories = []
        for category in expected_categories:
            category_path = os.path.join(dataset_root, category)
            
            if not os.path.exists(category_path):
                missing_categories.append(category)
                continue
                
            # Check train/test directories
            for split in ['train', 'test']:
                split_path = os.path.join(category_path, split)
                if not os.path.exists(split_path):
                    print(f"Warning: {split} directory missing in {category}!")
        
        if missing_categories:
            print("Missing categories:", ", ".join(missing_categories))
            return False
            
        print("Dataset verification completed successfully!")
        return True

    def get_dataset_stats(self, dataset_root):
        """Same as before..."""
        pass

    def plot_distribution(self, save_path=None):
        """Same as before..."""
        pass

def main():
    """Main function to download and prepare the dataset"""
    # Initialize downloader
    downloader = MVTecDownloader()
    
    # Download dataset
    dataset_root = downloader.download()
    
    # Verify dataset structure
    if not downloader.verify_dataset(dataset_root):
        print("Dataset verification failed!")
        sys.exit(1)
    
    # Collect and display statistics
    stats = downloader.get_dataset_stats(dataset_root)
    
    # Plot distribution
    downloader.plot_distribution(save_path="dataset_distribution.png")
    
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()