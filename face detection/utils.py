"""
Utility functions for face detection pipeline.
Handles image validation, loading, and data processing.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


def get_image_files(image_dir: str, supported_extensions: list = None) -> List[str]:
    """
    Retrieve all image files from a directory.

    Args:
        image_dir: Path to directory containing images
        supported_extensions: List of supported file extensions (default: jpg, png)

    Returns:
        Sorted list of absolute paths to image files
    """
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png']

    # Convert to lowercase for comparison
    supported_extensions = [ext.lower() for ext in supported_extensions]

    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Collect all image files recursively
    image_files = []
    for ext in supported_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    return [str(f) for f in image_files]


def load_image(image_path: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    Safely load an image file.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (image array or None, error flag)
        error flag = 1 if image failed to load, 0 if successful
    """
    try:
        # Read image using OpenCV (BGR format)
        image = cv2.imread(image_path)

        # Check if image loaded successfully
        if image is None:
            return None, 1

        # Verify image has valid dimensions
        if image.shape[0] == 0 or image.shape[1] == 0:
            return None, 1

        return image, 0
    except Exception as e:
        # Catch any unexpected errors during image loading
        print(f"  Warning: Error loading {os.path.basename(image_path)}: {str(e)}")
        return None, 1


def validate_output_directory(output_dir: str = 'output') -> str:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Path to output directory

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def format_results_summary(
    total_images: int,
    images_with_faces: int,
    total_errors: int
) -> str:
    """
    Format a summary of face detection results for logging.

    Args:
        total_images: Total images processed
        images_with_faces: Number of images with at least one face
        total_errors: Number of images that failed to process

    Returns:
        Formatted summary string
    """
    if total_images == 0:
        return "No images processed."

    pct_with_faces = (images_with_faces / (total_images - total_errors) * 100) if (total_images - total_errors) > 0 else 0
    pct_errors = (total_errors / total_images * 100)

    summary = f"""
    =============== FACE DETECTION SUMMARY ===============
    Total images processed: {total_images}
    Images with at least one face: {images_with_faces} ({pct_with_faces:.1f}%)
    Images with no faces detected: {total_images - total_errors - images_with_faces} ({100 - pct_with_faces - pct_errors:.1f}%)
    Processing errors: {total_errors} ({pct_errors:.1f}%)
    ====================================================
    """
    return summary
