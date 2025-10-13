"""
Coordinate Processing Utilities

This module provides utilities for transforming and processing coordinates
used in EAST text detection, including conversions between different
coordinate systems and geometric operations on text instances.
"""

import numpy as np
import cv2
from typing import List, Tuple, Union, Optional
import math
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """Axis-aligned bounding box representation"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def width(self) -> float:
        """Get bounding box width"""
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        """Get bounding box height"""
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center point"""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    @property
    def area(self) -> float:
        """Get bounding box area"""
        return max(0, self.width * self.height)

@dataclass
class OrientedBoundingBox:
    """Oriented bounding box representation (RBOX)"""
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float  # Rotation angle in radians
    
    def to_quad(self) -> np.ndarray:
        """Convert oriented bounding box to quadrilateral points"""
        # Half dimensions
        w_half = self.width / 2
        h_half = self.height / 2
        
        # Corner points in local coordinate system
        corners = np.array([
            [-w_half, -h_half],
            [w_half, -h_half], 
            [w_half, h_half],
            [-w_half, h_half]
        ])
        
        # Rotation matrix
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Rotate and translate
        rotated_corners = corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([self.center_x, self.center_y])
        
        return translated_corners


class CoordinateProcessor:
    """Main class for coordinate processing operations"""
    
    def __init__(self):
        pass
    
    def quad_to_bbox(self, quad: np.ndarray) -> BoundingBox:
        """
        Convert quadrilateral to axis-aligned bounding box
        
        Args:
            quad: Quadrilateral points as (4, 2) array
            
        Returns:
            BoundingBox object
        """
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]
        
        return BoundingBox(
            x_min=float(np.min(x_coords)),
            y_min=float(np.min(y_coords)),
            x_max=float(np.max(x_coords)),
            y_max=float(np.max(y_coords))
        )
    
    def quad_to_rbox(self, quad: np.ndarray) -> OrientedBoundingBox:
        """
        Convert quadrilateral to oriented bounding box (RBOX)
        
        Args:
            quad: Quadrilateral points as (4, 2) array in clockwise order
            
        Returns:
            OrientedBoundingBox object
        """
        # Calculate center
        center = np.mean(quad, axis=0)
        
        # Find the dominant orientation using the longest edge
        edge_vectors = []
        edge_lengths = []
        
        for i in range(4):
            start = quad[i]
            end = quad[(i + 1) % 4]
            vector = end - start
            length = np.linalg.norm(vector)
            
            edge_vectors.append(vector)
            edge_lengths.append(length)
        
        # Use the longest edge to determine orientation
        longest_edge_idx = np.argmax(edge_lengths)
        primary_vector = edge_vectors[longest_edge_idx]
        
        # Calculate angle of the primary edge
        angle = math.atan2(primary_vector[1], primary_vector[0])
        
        # Normalize angle to [-π/2, π/2] for text orientation
        if angle > math.pi / 2:
            angle -= math.pi
        elif angle < -math.pi / 2:
            angle += math.pi
        
        # Calculate dimensions by projecting quad onto oriented axes
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Project points onto the oriented axes
        projected_x = []
        projected_y = []
        
        for point in quad:
            # Translate to center-relative coordinates
            rel_point = point - center
            
            # Project onto rotated axes
            proj_x = rel_point[0] * cos_a + rel_point[1] * sin_a
            proj_y = -rel_point[0] * sin_a + rel_point[1] * cos_a
            
            projected_x.append(proj_x)
            projected_y.append(proj_y)
        
        # Calculate dimensions
        width = max(projected_x) - min(projected_x)
        height = max(projected_y) - min(projected_y)
        
        return OrientedBoundingBox(
            center_x=float(center[0]),
            center_y=float(center[1]),
            width=float(width),
            height=float(height),
            angle=float(angle)
        )
    
    def normalize_coordinates(self, 
                             coords: Union[np.ndarray, List], 
                             img_width: int, 
                             img_height: int) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range
        
        Args:
            coords: Coordinates to normalize
            img_width: Image width
            img_height: Image height
            
        Returns:
            Normalized coordinates as numpy array
        """
        coords = np.array(coords)
        original_shape = coords.shape
        
        # Reshape to handle both flat and structured coordinates
        if coords.ndim == 1 and len(coords) == 8:
            # Flat coordinate list [x1,y1,x2,y2,x3,y3,x4,y4]
            coords = coords.reshape(4, 2)
        
        normalized = coords.copy().astype(float)
        normalized[:, 0] /= img_width   # Normalize x coordinates
        normalized[:, 1] /= img_height  # Normalize y coordinates
        
        # Clip to [0, 1] range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            normalized = normalized.flatten()
        
        return normalized
    
    def denormalize_coordinates(self, 
                               coords: Union[np.ndarray, List], 
                               img_width: int, 
                               img_height: int) -> np.ndarray:
        """
        Denormalize coordinates from [0, 1] range to image coordinates
        
        Args:
            coords: Normalized coordinates
            img_width: Target image width
            img_height: Target image height
            
        Returns:
            Denormalized coordinates as numpy array
        """
        coords = np.array(coords)
        original_shape = coords.shape
        
        # Reshape to handle both flat and structured coordinates
        if coords.ndim == 1 and len(coords) == 8:
            coords = coords.reshape(4, 2)
        
        denormalized = coords.copy().astype(float)
        denormalized[:, 0] *= img_width   # Denormalize x coordinates
        denormalized[:, 1] *= img_height  # Denormalize y coordinates
        
        # Round to integer pixel coordinates
        denormalized = np.round(denormalized).astype(int)
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            denormalized = denormalized.flatten()
        
        return denormalized
    
    def resize_coordinates(self, 
                          coords: Union[np.ndarray, List],
                          original_size: Tuple[int, int],
                          target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize coordinates for image scaling
        
        Args:
            coords: Original coordinates
            original_size: Original image size (width, height)
            target_size: Target image size (width, height)
            
        Returns:
            Resized coordinates
        """
        coords = np.array(coords)
        original_shape = coords.shape
        
        # Calculate scaling factors
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        
        # Reshape if needed
        if coords.ndim == 1 and len(coords) == 8:
            coords = coords.reshape(4, 2)
        
        resized = coords.copy().astype(float)
        resized[:, 0] *= scale_x  # Scale x coordinates
        resized[:, 1] *= scale_y  # Scale y coordinates
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            resized = resized.flatten()
        
        return resized
    
    def calculate_quad_area(self, quad: np.ndarray) -> float:
        """
        Calculate area of quadrilateral using shoelace formula
        
        Args:
            quad: Quadrilateral points as (4, 2) array
            
        Returns:
            Area of the quadrilateral
        """
        x = quad[:, 0]
        y = quad[:, 1]
        
        # Shoelace formula
        area = 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
        return area
    
    def is_valid_quad(self, quad: np.ndarray, min_area: float = 10.0) -> bool:
        """
        Check if quadrilateral is valid (non-degenerate, sufficient area)
        
        Args:
            quad: Quadrilateral points as (4, 2) array
            min_area: Minimum area threshold
            
        Returns:
            True if quadrilateral is valid
        """
        # Check area
        area = self.calculate_quad_area(quad)
        if area < min_area:
            return False
        
        # Check for degenerate cases (collinear points)
        for i in range(4):
            p1 = quad[i]
            p2 = quad[(i + 1) % 4]
            p3 = quad[(i + 2) % 4]
            
            # Calculate cross product to check collinearity
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            
            if abs(cross) < 1e-6:  # Nearly collinear
                return False
        
        return True
    
    def ensure_clockwise_order(self, quad: np.ndarray) -> np.ndarray:
        """
        Ensure quadrilateral points are in clockwise order
        
        Args:
            quad: Quadrilateral points as (4, 2) array
            
        Returns:
            Reordered quadrilateral points
        """
        # Calculate center
        center = np.mean(quad, axis=0)
        
        # Calculate angles from center to each point
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        
        # Sort by angle (descending for clockwise)
        sorted_indices = np.argsort(-angles)
        
        return quad[sorted_indices]
    
    def clip_quad_to_image(self, 
                          quad: np.ndarray, 
                          img_width: int, 
                          img_height: int) -> np.ndarray:
        """
        Clip quadrilateral coordinates to image boundaries
        
        Args:
            quad: Quadrilateral points as (4, 2) array
            img_width: Image width
            img_height: Image height
            
        Returns:
            Clipped quadrilateral points
        """
        clipped = quad.copy()
        
        # Clip x coordinates
        clipped[:, 0] = np.clip(clipped[:, 0], 0, img_width - 1)
        
        # Clip y coordinates  
        clipped[:, 1] = np.clip(clipped[:, 1], 0, img_height - 1)
        
        return clipped


# Convenience functions for common operations
def quad_to_bbox(quad: np.ndarray) -> BoundingBox:
    """Convenience function for quadrilateral to bounding box conversion"""
    processor = CoordinateProcessor()
    return processor.quad_to_bbox(quad)

def quad_to_rbox(quad: np.ndarray) -> OrientedBoundingBox:
    """Convenience function for quadrilateral to oriented bounding box conversion"""
    processor = CoordinateProcessor()
    return processor.quad_to_rbox(quad)

def normalize_quad(quad: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    """Convenience function for coordinate normalization"""
    processor = CoordinateProcessor()
    return processor.normalize_coordinates(quad, img_width, img_height)