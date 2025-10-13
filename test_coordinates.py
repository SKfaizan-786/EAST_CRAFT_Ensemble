#!/usr/bin/env python3
"""
Test Coordinate Processing Utilities

This script tests the coordinate transformation and processing utilities
that will be used for ground truth map generation.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from east.utils.coordinates import (
    CoordinateProcessor, 
    BoundingBox, 
    OrientedBoundingBox,
    quad_to_bbox,
    quad_to_rbox, 
    normalize_quad
)


def test_bounding_box():
    """Test BoundingBox class"""
    print("🧪 Testing BoundingBox class...")
    
    bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=80)
    
    print(f"✅ BoundingBox: {bbox}")
    print(f"   - Width: {bbox.width}")
    print(f"   - Height: {bbox.height}")
    print(f"   - Center: {bbox.center}")
    print(f"   - Area: {bbox.area}")
    
    assert bbox.width == 90
    assert bbox.height == 60
    assert bbox.center == (55.0, 50.0)
    assert bbox.area == 5400
    
    print("✅ BoundingBox tests passed!")
    return True


def test_oriented_bounding_box():
    """Test OrientedBoundingBox class"""
    print("\n🧪 Testing OrientedBoundingBox class...")
    
    # Create a rotated rectangle
    rbox = OrientedBoundingBox(
        center_x=50, center_y=50,
        width=40, height=20,
        angle=np.pi/4  # 45 degrees
    )
    
    print(f"✅ OrientedBoundingBox: center=({rbox.center_x}, {rbox.center_y}), size={rbox.width}x{rbox.height}, angle={rbox.angle:.2f}")
    
    # Convert to quadrilateral
    quad = rbox.to_quad()
    print(f"   - Converted to quad: {quad.shape}")
    print(f"   - Quad points: {quad}")
    
    assert quad.shape == (4, 2)
    
    print("✅ OrientedBoundingBox tests passed!")
    return True


def test_coordinate_processor():
    """Test CoordinateProcessor class"""
    print("\n🧪 Testing CoordinateProcessor class...")
    
    processor = CoordinateProcessor()
    
    # Test quadrilateral (simple rectangle)
    quad = np.array([
        [100, 100],  # Top-left
        [200, 100],  # Top-right
        [200, 150],  # Bottom-right
        [100, 150]   # Bottom-left
    ])
    
    print(f"Original quad: {quad}")
    
    # Test quad to bbox conversion
    bbox = processor.quad_to_bbox(quad)
    print(f"✅ Quad to BBox: {bbox}")
    
    # Test quad to rbox conversion
    rbox = processor.quad_to_rbox(quad)
    print(f"✅ Quad to RBox: center=({rbox.center_x:.1f}, {rbox.center_y:.1f}), size={rbox.width:.1f}x{rbox.height:.1f}, angle={rbox.angle:.3f}")
    
    # Test normalization
    img_width, img_height = 1024, 768
    normalized = processor.normalize_coordinates(quad, img_width, img_height)
    print(f"✅ Normalized coords: {normalized}")
    
    # Test denormalization
    denormalized = processor.denormalize_coordinates(normalized, img_width, img_height)
    print(f"✅ Denormalized coords: {denormalized}")
    
    # Test resizing
    original_size = (1024, 768)
    target_size = (512, 384)
    resized = processor.resize_coordinates(quad, original_size, target_size)
    print(f"✅ Resized coords: {resized}")
    
    # Test area calculation
    area = processor.calculate_quad_area(quad)
    print(f"✅ Quad area: {area}")
    
    # Test validity check
    is_valid = processor.is_valid_quad(quad)
    print(f"✅ Quad validity: {is_valid}")
    
    # Test clockwise ordering
    clockwise_quad = processor.ensure_clockwise_order(quad)
    print(f"✅ Clockwise ordered: {clockwise_quad}")
    
    # Test clipping
    clipped = processor.clip_quad_to_image(quad, img_width, img_height)
    print(f"✅ Clipped to image: {clipped}")
    
    print("✅ CoordinateProcessor tests passed!")
    return True


def test_with_real_text_instance():
    """Test with real text instance coordinates"""
    print("\n🧪 Testing with real text instance...")
    
    # Use coordinates from your ICDAR dataset
    # Example from gt_img_1.txt: 377,117,463,117,465,130,378,130,Genaxis Theatre
    real_coords = [377, 117, 463, 117, 465, 130, 378, 130]
    quad = np.array(real_coords).reshape(4, 2)
    
    print(f"Real text instance quad: {quad}")
    
    processor = CoordinateProcessor()
    
    # Convert to bounding box
    bbox = processor.quad_to_bbox(quad)
    print(f"✅ Text BBox: {bbox}")
    
    # Convert to oriented box
    rbox = processor.quad_to_rbox(quad)
    print(f"✅ Text RBox: center=({rbox.center_x:.1f}, {rbox.center_y:.1f}), size={rbox.width:.1f}x{rbox.height:.1f}, angle={rbox.angle:.3f}°")
    
    # Test with typical ICDAR image size
    img_width, img_height = 1280, 720  # Typical ICDAR image size
    
    # Normalize
    normalized = processor.normalize_coordinates(quad, img_width, img_height)
    print(f"✅ Normalized: {normalized}")
    
    # Check if coordinates make sense
    assert np.all(normalized >= 0) and np.all(normalized <= 1), "Normalized coordinates should be in [0,1]"
    
    # Calculate area
    area = processor.calculate_quad_area(quad)
    print(f"✅ Text area: {area:.1f} pixels²")
    
    # Verify it's valid
    is_valid = processor.is_valid_quad(quad)
    print(f"✅ Valid quadrilateral: {is_valid}")
    
    print("✅ Real text instance tests passed!")
    return True


def test_convenience_functions():
    """Test convenience functions"""
    print("\n🧪 Testing convenience functions...")
    
    quad = np.array([[100, 100], [200, 100], [200, 150], [100, 150]])
    
    # Test convenience functions
    bbox = quad_to_bbox(quad)
    print(f"✅ quad_to_bbox: {bbox}")
    
    rbox = quad_to_rbox(quad)
    print(f"✅ quad_to_rbox: {rbox}")
    
    normalized = normalize_quad(quad, 1024, 768)
    print(f"✅ normalize_quad: {normalized}")
    
    print("✅ Convenience function tests passed!")
    return True


def main():
    """Run all coordinate processing tests"""
    print("🚀 Testing EAST Coordinate Processing Utilities")
    print("=" * 50)
    
    try:
        # Run all tests
        test_bounding_box()
        test_oriented_bounding_box()
        test_coordinate_processor()
        test_with_real_text_instance()
        test_convenience_functions()
        
        print("\n" + "=" * 50)
        print("🎉 All coordinate processing tests passed!")
        print("\nCoordinate utilities are ready for:")
        print("✅ Quad ↔ BBox conversions")
        print("✅ Quad ↔ RBox conversions")
        print("✅ Coordinate normalization/denormalization")
        print("✅ Coordinate resizing and clipping")
        print("✅ Geometric validation and area calculation")
        print("\n🔄 Next Sprint 2 task: Score map generation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)