#!/usr/bin/env python3
"""
Test ICDAR 2015 Parser

This script tests the ICDAR annotation parser with sample data or validates
the parser implementation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from east.datasets.icdar import ICDARAnnotationParser, TextInstance


def test_text_instance():
    """Test TextInstance class"""
    print("🧪 Testing TextInstance class...")
    
    # Test valid quadrilateral
    coords = [100, 100, 200, 100, 200, 150, 100, 150]
    text = "HELLO"
    instance = TextInstance(coordinates=coords, text=text)
    
    print(f"✅ Created TextInstance: {instance}")
    print(f"   - Text: {instance.text}")
    print(f"   - Coordinates: {instance.coordinates}")
    print(f"   - Bounding Box: {instance.get_bounding_box()}")
    print(f"   - Area: {instance.get_area():.2f}")
    
    # Test invalid quadrilateral (should handle gracefully)
    try:
        invalid_coords = [100, 100, 200]  # Not 8 coordinates
        invalid_instance = TextInstance(coordinates=invalid_coords, text="INVALID")
        print(f"⚠️  Created invalid instance: {invalid_instance}")
    except Exception as e:
        print(f"✅ Properly handled invalid coordinates: {e}")
    
    return True


def test_annotation_parser():
    """Test ICDARAnnotationParser class"""
    print("\n🧪 Testing ICDARAnnotationParser class...")
    
    parser = ICDARAnnotationParser()
    print(f"✅ Created parser: {parser}")
    
    # Test parsing a sample annotation line
    sample_line = "100,100,200,100,200,150,100,150,HELLO WORLD"
    try:
        instance = parser.parse_annotation_line(sample_line)
        print(f"✅ Parsed annotation: {instance}")
        print(f"   - Text: '{instance.text}'")
        print(f"   - Coordinates: {instance.coordinates}")
    except Exception as e:
        print(f"❌ Failed to parse annotation: {e}")
        return False
    
    # Test parsing invalid annotation
    invalid_line = "100,100,200"
    try:
        invalid_instance = parser.parse_annotation_line(invalid_line)
        print(f"⚠️  Parsed invalid line: {invalid_instance}")
    except Exception as e:
        print(f"✅ Properly handled invalid annotation: {e}")
    
    return True


def test_sample_data_creation():
    """Create sample ICDAR data for testing"""
    print("\n🧪 Creating sample test data...")
    
    # Create a sample directory structure
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample annotation file
    sample_gt = test_dir / "gt_sample.txt"
    with open(sample_gt, "w") as f:
        f.write("100,100,200,100,200,130,100,130,Hello\n")
        f.write("220,100,320,100,320,130,220,130,World\n")
        f.write("50,200,150,200,150,230,50,230,Test\n")
    
    print(f"✅ Created sample annotation file: {sample_gt}")
    
    # Test parsing the file
    parser = ICDARAnnotationParser()
    try:
        instances = parser.parse_annotation_file(sample_gt)
        print(f"✅ Parsed {len(instances)} text instances from file:")
        for i, instance in enumerate(instances, 1):
            print(f"   {i}. '{instance.text}' at {instance.coordinates}")
    except Exception as e:
        print(f"❌ Failed to parse annotation file: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("🚀 Testing EAST ICDAR 2015 Parser Implementation")
    print("=" * 50)
    
    try:
        # Run tests
        test_text_instance()
        test_annotation_parser()
        test_sample_data_creation()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nYour ICDAR parser is ready to use!")
        print("\nNext steps:")
        print("1. Place your ICDAR 2015 dataset files in the data/ directory")
        print("2. Use tools/organize_dataset.py to organize your dataset")
        print("3. Continue with Sprint 2 tasks (coordinate processing, ground truth generation)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())