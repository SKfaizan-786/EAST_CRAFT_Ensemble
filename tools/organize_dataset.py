"""
ICDAR Dataset Organization Helper

Helps organize manually downloaded ICDAR 2015 files into the expected structure
and creates train/validation splits.
"""

import os
import shutil
import zipfile
from pathlib import Path
import logging
from typing import List, Dict
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICDARDatasetOrganizer:
    """Helper to organize ICDAR 2015 dataset files"""
    
    def __init__(self, target_dir: str = "data/icdar2015"):
        self.target_dir = Path(target_dir)
        self.ensure_directory_structure()
    
    def ensure_directory_structure(self):
        """Create the expected directory structure"""
        dirs_to_create = [
            'train/images', 'train/annotations',
            'test/images', 'test/annotations', 
            'splits'
        ]
        
        for dir_path in dirs_to_create:
            (self.target_dir / dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"✅ Directory structure created at {self.target_dir}")
    
    def extract_and_organize_zip(self, zip_path: str, file_type: str):
        """
        Extract and organize files from ICDAR zip
        
        Args:
            zip_path: Path to the zip file
            file_type: 'train_images', 'train_gt', 'test_images', or 'test_gt'
        """
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            logger.error(f"❌ Zip file not found: {zip_path}")
            return False
        
        logger.info(f"📦 Extracting {zip_path}")
        
        # Extract to temporary directory
        temp_dir = self.target_dir / "temp_extract"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Organize files based on type
            if file_type == 'train_images':
                self._organize_train_images(temp_dir)
            elif file_type == 'train_gt':
                self._organize_train_annotations(temp_dir)
            elif file_type == 'test_images':
                self._organize_test_images(temp_dir)
            elif file_type == 'test_gt':
                self._organize_test_annotations(temp_dir)
            else:
                logger.error(f"❌ Unknown file type: {file_type}")
                return False
            
            logger.info(f"✅ Organized {file_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to extract {zip_path}: {e}")
            return False
        
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _organize_train_images(self, extract_dir: Path):
        """Organize training images"""
        target_dir = self.target_dir / 'train' / 'images'
        self._move_images(extract_dir, target_dir)
    
    def _organize_train_annotations(self, extract_dir: Path):
        """Organize training annotations"""
        target_dir = self.target_dir / 'train' / 'annotations'
        self._move_annotations(extract_dir, target_dir)
    
    def _organize_test_images(self, extract_dir: Path):
        """Organize test images"""  
        target_dir = self.target_dir / 'test' / 'images'
        self._move_images(extract_dir, target_dir)
    
    def _organize_test_annotations(self, extract_dir: Path):
        """Organize test annotations"""
        target_dir = self.target_dir / 'test' / 'annotations'
        self._move_annotations(extract_dir, target_dir)
    
    def _move_images(self, source_dir: Path, target_dir: Path):
        """Move image files to target directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        moved_count = 0
        
        for file_path in source_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                target_path = target_dir / file_path.name
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
        
        logger.info(f"📷 Moved {moved_count} images to {target_dir}")
    
    def _move_annotations(self, source_dir: Path, target_dir: Path):
        """Move annotation files to target directory"""
        moved_count = 0
        
        for file_path in source_dir.rglob('*.txt'):
            if file_path.is_file():
                target_path = target_dir / file_path.name
                shutil.move(str(file_path), str(target_path))
                moved_count += 1
        
        logger.info(f"📝 Moved {moved_count} annotation files to {target_dir}")
    
    def organize_manual_files(self, 
                            train_images_dir: str = None,
                            train_annotations_dir: str = None,
                            test_images_dir: str = None,
                            test_annotations_dir: str = None):
        """
        Organize manually placed files
        
        Args:
            train_images_dir: Directory containing training images
            train_annotations_dir: Directory containing training annotations
            test_images_dir: Directory containing test images  
            test_annotations_dir: Directory containing test annotations
        """
        logger.info("📁 Organizing manually placed files...")
        
        if train_images_dir:
            self._copy_files(train_images_dir, self.target_dir / 'train' / 'images', ['.jpg', '.png'])
        
        if train_annotations_dir:
            self._copy_files(train_annotations_dir, self.target_dir / 'train' / 'annotations', ['.txt'])
        
        if test_images_dir:
            self._copy_files(test_images_dir, self.target_dir / 'test' / 'images', ['.jpg', '.png'])
        
        if test_annotations_dir:
            self._copy_files(test_annotations_dir, self.target_dir / 'test' / 'annotations', ['.txt'])
        
        logger.info("✅ Manual file organization complete")
    
    def _copy_files(self, source_dir: str, target_dir: Path, extensions: List[str]):
        """Copy files with specified extensions"""
        source_path = Path(source_dir)
        
        if not source_path.exists():
            logger.warning(f"⚠️  Source directory not found: {source_dir}")
            return
        
        copied_count = 0
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                target_path = target_dir / file_path.name
                if not target_path.exists():
                    shutil.copy2(str(file_path), str(target_path))
                    copied_count += 1
        
        logger.info(f"📋 Copied {copied_count} files to {target_dir}")
    
    def create_train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        """Create train/validation split"""
        logger.info(f"🔄 Creating train/val split (val_ratio={val_ratio})")
        
        train_images_dir = self.target_dir / 'train' / 'images'
        
        if not train_images_dir.exists() or not any(train_images_dir.iterdir()):
            logger.error(f"❌ No training images found in {train_images_dir}")
            return False
        
        # Get image file stems (without extension)
        image_files = []
        for img_path in train_images_dir.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_files.append(img_path.stem)
        
        if not image_files:
            logger.error("❌ No valid image files found")
            return False
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(image_files)
        
        val_size = int(len(image_files) * val_ratio)
        val_files = image_files[:val_size]
        train_files = image_files[val_size:]
        
        # Write split files
        splits_dir = self.target_dir / 'splits'
        
        with open(splits_dir / 'train.txt', 'w') as f:
            for filename in train_files:
                f.write(f"{filename}\n")
        
        with open(splits_dir / 'val.txt', 'w') as f:
            for filename in val_files:
                f.write(f"{filename}\n")
        
        logger.info(f"✅ Created splits: {len(train_files)} train, {len(val_files)} val")
        return True
    
    def validate_organization(self) -> Dict:
        """Validate the organized dataset"""
        logger.info("🔍 Validating dataset organization...")
        
        results = {
            'valid': True,
            'errors': [],
            'statistics': {}
        }
        
        # Check directories exist
        required_dirs = [
            'train/images', 'train/annotations',
            'test/images', 'test/annotations',
            'splits'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.target_dir / dir_name
            if not dir_path.exists():
                results['valid'] = False
                results['errors'].append(f"Missing directory: {dir_name}")
        
        if not results['valid']:
            return results
        
        # Count files
        train_images = len(list((self.target_dir / 'train' / 'images').glob('*.[jp][pn]g')))
        train_annotations = len(list((self.target_dir / 'train' / 'annotations').glob('*.txt')))
        test_images = len(list((self.target_dir / 'test' / 'images').glob('*.[jp][pn]g')))
        test_annotations = len(list((self.target_dir / 'test' / 'annotations').glob('*.txt')))
        
        results['statistics'] = {
            'train_images': train_images,
            'train_annotations': train_annotations,
            'test_images': test_images,
            'test_annotations': test_annotations
        }
        
        # Check split files
        train_split = self.target_dir / 'splits' / 'train.txt'
        val_split = self.target_dir / 'splits' / 'val.txt'
        
        if train_split.exists() and val_split.exists():
            with open(train_split) as f:
                train_split_count = len(f.readlines())
            with open(val_split) as f:
                val_split_count = len(f.readlines())
                
            results['statistics']['train_split'] = train_split_count
            results['statistics']['val_split'] = val_split_count
        
        return results


def main():
    """Interactive dataset organization"""
    print("🎯 ICDAR 2015 Dataset Organizer")
    print("=" * 50)
    
    organizer = ICDARDatasetOrganizer()
    
    print("\nChoose organization method:")
    print("1. Extract from ZIP files")
    print("2. Organize manually placed files") 
    print("3. Just create train/val splits")
    print("4. Validate existing organization")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n📦 ZIP File Organization")
        print("Expected files:")
        print("- ch4_training_images.zip")
        print("- ch4_training_localization_transcription_gt.zip") 
        print("- ch4_test_images.zip")
        print("- Challenge4_Test_Task1_GT.zip")
        
        zip_mappings = {
            'train_images': input("Path to training images ZIP: ").strip(),
            'train_gt': input("Path to training GT ZIP: ").strip(),
            'test_images': input("Path to test images ZIP (optional): ").strip(),
            'test_gt': input("Path to test GT ZIP (optional): ").strip()
        }
        
        for file_type, zip_path in zip_mappings.items():
            if zip_path:
                organizer.extract_and_organize_zip(zip_path, file_type)
    
    elif choice == "2":
        print("\n📁 Manual File Organization")
        organizer.organize_manual_files(
            train_images_dir=input("Training images directory: ").strip() or None,
            train_annotations_dir=input("Training annotations directory: ").strip() or None,
            test_images_dir=input("Test images directory (optional): ").strip() or None,
            test_annotations_dir=input("Test annotations directory (optional): ").strip() or None
        )
    
    elif choice == "3":
        print("\n🔄 Creating train/val splits only")
        
    elif choice == "4":
        print("\n🔍 Validating existing organization")
        
    else:
        print("❌ Invalid choice")
        return
    
    # Always create splits and validate
    if choice in ["1", "2", "3"]:
        organizer.create_train_val_split()
    
    # Validate
    results = organizer.validate_organization()
    
    print(f"\n📊 Validation Results:")
    if results['valid']:
        print("✅ Dataset organization is valid")
        stats = results['statistics']
        print(f"📷 Train images: {stats.get('train_images', 0)}")
        print(f"📝 Train annotations: {stats.get('train_annotations', 0)}")
        print(f"🧪 Test images: {stats.get('test_images', 0)}")
        print(f"📄 Test annotations: {stats.get('test_annotations', 0)}")
        if 'train_split' in stats:
            print(f"📋 Train split: {stats['train_split']} files")
            print(f"📋 Val split: {stats['val_split']} files")
    else:
        print("❌ Validation failed:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\n🎉 Dataset organization complete!")
    print(f"📁 Location: {organizer.target_dir}")


if __name__ == "__main__":
    main()