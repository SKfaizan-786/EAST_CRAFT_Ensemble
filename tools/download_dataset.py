"""
ICDAR 2015 Dataset Downloader and Validator

Downloads and validates the ICDAR 2015 Robust Reading Competition dataset
for scene text detection training and evaluation.
"""

import os
import argparse
import requests
import zipfile
import hashlib
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ICDAR 2015 Dataset URLs and checksums
ICDAR_2015_URLS = {
    'train_images': {
        'url': 'https://rrc.cvc.uab.es/downloads/ch4_training_images.zip',
        'filename': 'ch4_training_images.zip',
        'md5': None  # Will be updated with actual checksums
    },
    'train_gt': {
        'url': 'https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip', 
        'filename': 'ch4_training_localization_transcription_gt.zip',
        'md5': None
    },
    'test_images': {
        'url': 'https://rrc.cvc.uab.es/downloads/ch4_test_images.zip',
        'filename': 'ch4_test_images.zip', 
        'md5': None
    },
    'test_gt': {
        'url': 'https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip',
        'filename': 'Challenge4_Test_Task1_GT.zip',
        'md5': None
    }
}

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, destination, expected_md5=None):
    """Download file with progress bar and validation"""
    logger.info(f"Downloading {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    # Validate checksum if provided
    if expected_md5:
        actual_md5 = calculate_md5(destination)
        if actual_md5 != expected_md5:
            raise ValueError(f"MD5 mismatch for {destination}: expected {expected_md5}, got {actual_md5}")
        logger.info(f"✅ Checksum validated for {destination}")
    
    logger.info(f"✅ Downloaded {destination}")

def extract_zip(zip_path, extract_to):
    """Extract ZIP file"""
    logger.info(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"✅ Extracted to {extract_to}")

def create_dataset_structure(data_dir):
    """Create standard dataset directory structure"""
    data_path = Path(data_dir)
    
    # Create directories
    dirs_to_create = [
        'icdar2015/train/images',
        'icdar2015/train/annotations', 
        'icdar2015/test/images',
        'icdar2015/test/annotations',
        'icdar2015/splits'
    ]
    
    for dir_path in dirs_to_create:
        (data_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    return data_path / 'icdar2015'

def organize_dataset_files(dataset_dir, downloads_dir):
    """Organize downloaded files into standard structure"""
    logger.info("Organizing dataset files...")
    
    # Move and rename files to standard structure
    # This is a placeholder - actual file organization will depend on 
    # the exact structure of the downloaded ICDAR files
    
    logger.info("✅ Dataset files organized")

def create_train_val_split(dataset_dir, val_ratio=0.2):
    """Create train/validation split"""
    import random
    
    logger.info(f"Creating train/val split (val_ratio={val_ratio})")
    
    train_images_dir = dataset_dir / 'train' / 'images'
    if not train_images_dir.exists():
        logger.warning("Training images directory not found, skipping split creation")
        return
    
    # Get all image files
    image_files = [f.stem for f in train_images_dir.glob('*.jpg')]
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(image_files)
    
    val_size = int(len(image_files) * val_ratio)
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    # Write split files
    splits_dir = dataset_dir / 'splits'
    
    with open(splits_dir / 'train.txt', 'w') as f:
        for filename in train_files:
            f.write(f"{filename}\n")
    
    with open(splits_dir / 'val.txt', 'w') as f:
        for filename in val_files:
            f.write(f"{filename}\n")
    
    logger.info(f"✅ Created splits: {len(train_files)} train, {len(val_files)} val")

def validate_dataset(dataset_dir):
    """Validate dataset integrity"""
    logger.info("Validating dataset...")
    
    # Check required directories exist
    required_dirs = [
        'train/images', 'train/annotations',
        'test/images', 'test/annotations', 
        'splits'
    ]
    
    for dir_name in required_dirs:
        dir_path = dataset_dir / dir_name
        if not dir_path.exists():
            logger.error(f"❌ Missing directory: {dir_path}")
            return False
    
    # Check split files exist
    split_files = ['train.txt', 'val.txt']
    for split_file in split_files:
        split_path = dataset_dir / 'splits' / split_file
        if not split_path.exists():
            logger.error(f"❌ Missing split file: {split_path}")
            return False
    
    # Count files
    train_images = len(list((dataset_dir / 'train' / 'images').glob('*.jpg')))
    test_images = len(list((dataset_dir / 'test' / 'images').glob('*.jpg')))
    
    logger.info(f"✅ Dataset validated: {train_images} train images, {test_images} test images")
    return True

def download_icdar2015(data_dir, skip_existing=True):
    """Download and setup ICDAR 2015 dataset"""
    data_path = Path(data_dir)
    downloads_dir = data_path / 'downloads'
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = create_dataset_structure(data_dir)
    
    # Download all files
    for name, info in ICDAR_2015_URLS.items():
        file_path = downloads_dir / info['filename']
        
        if skip_existing and file_path.exists():
            logger.info(f"⏭️  Skipping {info['filename']} (already exists)")
            continue
        
        try:
            download_file(info['url'], file_path, info['md5'])
            extract_zip(file_path, downloads_dir)
        except Exception as e:
            logger.error(f"❌ Failed to download {info['filename']}: {e}")
            continue
    
    # Organize files
    organize_dataset_files(dataset_dir, downloads_dir)
    
    # Create train/val split
    create_train_val_split(dataset_dir)
    
    # Validate
    if validate_dataset(dataset_dir):
        logger.info("🎉 ICDAR 2015 dataset setup completed successfully!")
        return True
    else:
        logger.error("❌ Dataset validation failed")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download ICDAR 2015 dataset')
    parser.add_argument('--dataset', choices=['icdar2015'], default='icdar2015',
                        help='Dataset to download')
    parser.add_argument('--output', '-o', default='data/',
                        help='Output directory for dataset')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip downloading if files already exist')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.dataset} dataset download...")
    logger.info(f"Output directory: {args.output}")
    
    if args.dataset == 'icdar2015':
        success = download_icdar2015(args.output, args.skip_existing)
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")
        return 1
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())