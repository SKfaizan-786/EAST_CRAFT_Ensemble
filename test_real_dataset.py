#!/usr/bin/env python3
"""
Test ICDAR Parser on Real Dataset

This script tests the ICDAR annotation parser on your actual ICDAR 2015 dataset.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from east.datasets.icdar import ICDARAnnotationParser, TextInstance


def test_real_dataset():
    """Test ICDAR parser with your actual dataset"""
    print("🚀 Testing ICDAR Parser on Your Real Dataset")
    print("=" * 50)
    
    # Use your actual data paths
    train_annotations_dir = Path("data/icdar2015/train/annotations")
    train_images_dir = Path("data/icdar2015/train/images")
    
    # Check if directories exist
    if not train_annotations_dir.exists():
        print(f"❌ Annotations directory not found: {train_annotations_dir}")
        return False
    
    if not train_images_dir.exists():
        print(f"❌ Images directory not found: {train_images_dir}")
        return False
    
    parser = ICDARAnnotationParser()
    
    # Get annotation files
    annotation_files = sorted([f for f in train_annotations_dir.glob("*.txt")])
    
    if not annotation_files:
        print("❌ No annotation files found!")
        return False
    
    print(f"✅ Found {len(annotation_files)} annotation files")
    
    # Test first few files
    success_count = 0
    error_count = 0
    
    for i, ann_file in enumerate(annotation_files[:5]):  # Test first 5 files
        print(f"\n--- Testing file {i+1}: {ann_file.name} ---")
        
        try:
            # Parse annotation file
            text_instances = parser.parse_annotation_file(ann_file)
            print(f"✅ Parsed {len(text_instances)} text instances")
            
            if text_instances:
                # Show first instance details
                instance = text_instances[0]
                print(f"   First instance: '{instance.text}' (difficult: {instance.difficult})")
                print(f"   Coordinates: {instance.coordinates}")
                print(f"   Valid quadrilateral: {instance.is_valid_quadrilateral()}")
                print(f"   Bounding box: {instance.get_bounding_box()}")
            
            success_count += 1
        
        except Exception as e:
            print(f"❌ Error parsing {ann_file.name}: {e}")
            error_count += 1
    
    print(f"\n--- Results Summary ---")
    print(f"✅ Successfully parsed: {success_count} files")
    print(f"❌ Errors: {error_count} files")
    
    # Validate dataset structure
    print(f"\n--- Dataset Structure Validation ---")
    try:
        is_valid = parser.validate_dataset_structure(
            images_dir=str(train_images_dir),
            annotations_dir=str(train_annotations_dir)
        )
        print(f"✅ Dataset structure validation: {'PASSED' if is_valid else 'FAILED'}")
    except Exception as e:
        print(f"❌ Dataset validation error: {e}")
        is_valid = False
    
    # Show dataset statistics
    if success_count > 0:
        print(f"\n--- Dataset Statistics ---")
        total_instances = 0
        difficult_instances = 0
        
        # Parse a few more files to get stats
        for ann_file in annotation_files[:20]:  # Sample 20 files
            try:
                instances = parser.parse_annotation_file(ann_file)
                total_instances += len(instances)
                difficult_instances += sum(1 for inst in instances if inst.difficult)
            except:
                continue
        
        print(f"Sample statistics (first 20 files):")
        print(f"  Total text instances: {total_instances}")
        print(f"  Difficult instances: {difficult_instances}")
        print(f"  Average per file: {total_instances/min(20, len(annotation_files)):.1f}")
    
    return success_count > error_count and is_valid


def peek_at_annotation_file():
    """Show contents of first annotation file for debugging"""
    print("\n🔍 Peeking at annotation file format...")
    
    ann_dir = Path("data/icdar2015/train/annotations")
    ann_files = list(ann_dir.glob("*.txt"))
    
    if ann_files:
        first_file = ann_files[0]
        print(f"Contents of {first_file.name}:")
        print("-" * 40)
        
        with open(first_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:5], 1):  # Show first 5 lines
                print(f"{i:2d}: {line.rstrip()}")
        
        if len(lines) > 5:
            print(f"... and {len(lines)-5} more lines")
        
        print("-" * 40)
        return True
    
    return False


if __name__ == "__main__":
    print("🔥 Testing Your ICDAR 2015 Dataset\n")
    
    # First, peek at the data format
    peek_at_annotation_file()
    
    # Then test the parser
    success = test_real_dataset()
    
    if success:
        print("\n🎉 SUCCESS! Your ICDAR parser is working perfectly with your dataset!")
        print("\nYour dataset is ready for Sprint 2 tasks:")
        print("✅ ICDAR annotation parsing")  
        print("🔄 Next: Coordinate processing utilities")
        print("🔄 Next: Score map generation")
        print("🔄 Next: Geometry map generation")
    else:
        print("\n⚠️ Some issues found. Check the error messages above.")
        print("Your dataset might need some adjustments.")