"""
RetinaFace Face Detection Pipeline for Social Media Research

This script detects human faces in images using the RetinaFace model
(Deng et al., 2020). For each image, it outputs:
  - face_presence: Binary indicator (1 = at least one face, 0 = no face)
  - face_count: Number of faces detected in the image
  - error_flag: Error indicator (1 = failed to process, 0 = success)

Usage:
    python detect_faces.py --image_dir /path/to/images
"""

import argparse
import csv
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore')

# Import custom utility functions
from utils import get_image_files, load_image, validate_output_directory, format_results_summary

# Import RetinaFace detector
try:
    from retinaface.RetinaFace import detect_faces
except ImportError:
    print("Error: retinaface package not installed.")
    print("Install with: pip install retina-face[pytorch]")
    sys.exit(1)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy and torch.

    Args:
        seed: Random seed value (default: 42)
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass


def get_retinaface_version() -> str:
    """
    Retrieve the RetinaFace package version.

    Returns:
        Version string
    """
    try:
        import retinaface
        return getattr(retinaface, '__version__', '0.0.17')
    except:
        return 'Unknown'


def detect_faces_in_image(image_array: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Detect faces in an image using RetinaFace model.

    The RetinaFace detect_faces function returns a dictionary where keys are
    face identifiers (e.g., "face_1", "face_2") and values contain face
    detection information (bounding boxes, landmarks, confidence scores).

    Args:
        image_array: Image as numpy array (BGR format from OpenCV)
        threshold: Detection confidence threshold (default: 0.5 = 50%)

    Returns:
        Dictionary with full detection data or empty dict if no faces found
    """
    try:
        # Run face detection on the image
        # Returns dict with detected faces: {"face_1": {...}, "face_2": {...}, ...}
        # or empty dict {} if no faces found
        detections = detect_faces(image_array, threshold=threshold)

        # Handle case where detections is None or empty
        if detections is None:
            return {}

        return detections
    except Exception as e:
        # If detection fails, return empty dict (will be handled by error_flag)
        return {}


def process_images(image_dir: str, output_csv: str = 'output/face_presence.csv',
                   output_summary_csv: str = 'output/face_summary.csv') -> None:
    """
    Main pipeline: detect faces in all images from a directory and save results.

    Args:
        image_dir: Path to directory containing images
        output_csv: Path to detailed face-level output CSV file
        output_summary_csv: Path to image-level summary CSV file
    """

    # Step 1: Set reproducibility seed
    set_random_seeds(seed=42)
    print(f"✓ Random seeds set for reproducibility")

    # Step 2: Initialize RetinaFace detector and display model info
    print(f"\n✓ Initializing RetinaFace detector...")
    model_version = get_retinaface_version()
    print(f"  RetinaFace version: {model_version}")
    print(f"  Citation: Deng et al., 2020")
    print(f"  Reference: RetinaFace: Single-stage Dense Face Localisation in the Wild")

    # Step 3: Collect image files from input directory
    print(f"\n✓ Scanning image directory: {image_dir}")
    try:
        image_files = get_image_files(image_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if len(image_files) == 0:
        print("Error: No .jpg or .png images found in directory")
        sys.exit(1)

    print(f"  Found {len(image_files)} images to process")

    # Step 4: Create output directory
    validate_output_directory(os.path.dirname(output_csv))

    # Step 5: Process images in batch with progress bar
    print(f"\n✓ Processing images...")
    results = []
    images_with_faces = 0
    total_errors = 0

    # Use tqdm for progress bar showing real-time processing status
    for image_path in tqdm(image_files, desc="Face Detection", unit="image"):
        image_filename = os.path.basename(image_path)

        # Attempt to load image
        image_array, error_flag = load_image(image_path)

        # If image failed to load, record error and continue to next image
        if error_flag == 1:
            results.append({
                'image_filename': image_filename,
                'face_id': None,
                'face_presence': 0,
                'face_count': 0,
                'bbox_x1': None,
                'bbox_y1': None,
                'bbox_x2': None,
                'bbox_y2': None,
                'confidence': None,
                'landmarks': None,
                'error_flag': 1
            })
            total_errors += 1
            continue

        # Run face detection on successfully loaded image
        # Using threshold of 0.5 (50% confidence) for balanced detection
        detections = detect_faces_in_image(image_array, threshold=0.5)

        # Convert to binary face_presence indicator
        face_presence = 1 if len(detections) > 0 else 0
        face_count = len(detections)

        # Track images with detected faces
        if face_presence == 1:
            images_with_faces += 1

        # If no faces detected, add single row
        if face_count == 0:
            results.append({
                'image_filename': image_filename,
                'face_id': None,
                'face_presence': 0,
                'face_count': 0,
                'bbox_x1': None,
                'bbox_y1': None,
                'bbox_x2': None,
                'bbox_y2': None,
                'confidence': None,
                'landmarks': None,
                'error_flag': 0
            })
        else:
            # Create one row per detected face with full details
            for face_id, face_data in detections.items():
                facial_area = face_data['facial_area']  # [x1, y1, x2, y2]
                landmarks = face_data['landmarks']  # dict with eye, nose, mouth coords
                confidence = face_data['score']

                # Format landmarks as string representation
                landmarks_str = str(landmarks) if landmarks else None

                results.append({
                    'image_filename': image_filename,
                    'face_id': face_id,
                    'face_presence': 1,
                    'face_count': face_count,
                    'bbox_x1': int(facial_area[0]),
                    'bbox_y1': int(facial_area[1]),
                    'bbox_x2': int(facial_area[2]),
                    'bbox_y2': int(facial_area[3]),
                    'confidence': round(float(confidence), 4),
                    'landmarks': landmarks_str,
                    'error_flag': 0
                })

    # Step 6: Write results to CSV files
    print(f"\n✓ Writing results to CSV files...")
    face_level_columns = ['image_filename', 'face_id', 'face_presence', 'face_count',
                          'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence', 'landmarks', 'error_flag']

    try:
        # Write face-level detailed CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=face_level_columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Face-level details saved to: {output_csv}")
    except Exception as e:
        print(f"Error writing face-level CSV: {e}")
        sys.exit(1)

    # Generate image-level summary (one row per image)
    try:
        image_summary = {}
        for result in results:
            img_name = result['image_filename']
            if img_name not in image_summary:
                image_summary[img_name] = {
                    'image_filename': img_name,
                    'face_presence': result['face_presence'],
                    'face_count': result['face_count'],
                    'error_flag': result['error_flag']
                }

        summary_rows = list(image_summary.values())
        summary_columns = ['image_filename', 'face_presence', 'face_count', 'error_flag']

        with open(output_summary_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=summary_columns)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Image-level summary saved to: {output_summary_csv}")
    except Exception:
        print(f"Error writing summary CSV")
        sys.exit(1)

    # Step 7: Print summary statistics
    summary = format_results_summary(len(image_files), images_with_faces, total_errors)
    print(summary)


def main():
    """
    Parse command-line arguments and run the face detection pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Detect faces in images using RetinaFace (Deng et al., 2020)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_faces.py --image_dir ./photos
  python detect_faces.py --image_dir /path/to/images --output_csv results/faces.csv
        """
    )

    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Path to directory containing .jpg and .png images'
    )

    parser.add_argument(
        '--output_csv',
        type=str,
        default='output/face_presence.csv',
        help='Path to output CSV file (default: output/face_presence.csv)'
    )

    args = parser.parse_args()

    # Validate that image directory exists
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        sys.exit(1)

    # Run the face detection pipeline
    print("=" * 60)
    print("RetinaFace Face Detection Pipeline")
    print("=" * 60)
    process_images(args.image_dir, args.output_csv)
    print("\n✓ Pipeline completed successfully!")


if __name__ == '__main__':
    main()