# RetinaFace Face Detection Pipeline

A modular Python pipeline for detecting human faces in images using the RetinaFace model (Deng et al., 2020). Designed for social media post engagement research.

## Overview

This pipeline processes images in a directory and outputs comprehensive facial detection data:
- **face_presence**: Binary indicator (1 = face present, 0 = no face)
- **face_count**: Total number of faces detected in the image
- **Bounding boxes**: Precise coordinates (x1, y1, x2, y2) for each detected face
- **Confidence scores**: Detection confidence level for each face (0-1 range)
- **Facial landmarks**: 5 key facial points (eyes, nose, mouth corners) for each face
- **Face ID**: Unique identifier for each detected face in multi-face images
- **error_flag**: Processing status (1 = failed, 0 = success)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
face detection % python detect_faces.py --image_dir "/Users/pouriakhansari/Library/CloudStorage/OneDrive-IveyBusinessSchool/Prashant/Content Format/Code/Feature Extraction/test_images"
```

### Optional Arguments
```bash
python detect_faces.py --image_dir /path/to/images --output_csv custom_output.csv
```

## Output

The pipeline generates **two CSV files** with complementary data:

### 1. Face-Level Details (`output/face_presence.csv`)
Detailed information with one row per detected face:

| Column | Type | Description |
|--------|------|-------------|
| image_filename | str | Name of the image file |
| face_id | str | Unique identifier for detected face (e.g., "face_1", "face_2") |
| face_presence | int | 1 if face detected, 0 otherwise |
| face_count | int | Total number of faces in the image |
| bbox_x1, bbox_y1 | int | Top-left corner coordinates of bounding box |
| bbox_x2, bbox_y2 | int | Bottom-right corner coordinates of bounding box |
| confidence | float | Detection confidence score (0-1) |
| landmarks | str | Dictionary with 5 facial landmarks (right_eye, left_eye, nose, mouth_right, mouth_left) |
| error_flag | int | 1 if processing failed, 0 if successful |

### 2. Image-Level Summary (`output/face_summary.csv`)
Aggregate statistics with one row per image:

| Column | Type | Description |
|--------|------|-------------|
| image_filename | str | Name of the image file |
| face_presence | int | 1 if face detected, 0 otherwise |
| face_count | int | Total number of faces in the image |
| error_flag | int | 1 if processing failed, 0 if successful |

## Example Output

**Face-Level CSV:**
```
image_filename,face_id,face_presence,face_count,bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,landmarks,error_flag
photo_1.jpg,face_1,1,2,445,613,663,905,0.9995,"{'right_eye': [510.6, 737.8], ...}",0
photo_1.jpg,face_2,1,2,232,4,686,472,0.9936,"{'right_eye': [441.7, 202.2], ...}",0
photo_3.jpg,face_1,1,1,397,62,607,358,0.9994,"{'right_eye': [460.4, 189.6], ...}",0
```

**Image-Level Summary CSV:**
```
image_filename,face_presence,face_count,error_flag
photo_1.jpg,1,2,0
photo_2.jpg,0,0,0
photo_3.jpg,1,1,0
```

## Features

- ✅ Binary face detection (presence/absence indicator)
- ✅ Face counting per image
- ✅ **NEW** Bounding box extraction for each detected face
- ✅ **NEW** Confidence scores for each detection
- ✅ **NEW** 5-point facial landmarks (eyes, nose, mouth)
- ✅ **NEW** Individual face tracking with unique IDs
- ✅ Robust error handling for corrupted/unreadable images
- ✅ Batch processing with progress bar (tqdm)
- ✅ Reproducible results with random seed control
- ✅ Well-documented code for academic researchers
- ✅ Detailed summary statistics after processing

## Supported Formats

- `.jpg` / `.jpeg`
- `.png`

## Citation

```bibtex
@inproceedings{deng2020retinaface,
  title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irini and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2020}
}
```

## Requirements

- Python 3.8+
- See requirements.txt for dependencies

## Project Structure

```
face detection/
├── detect_faces.py       # Main detection script
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .gitignore            # Git configuration
└── output/               # Results directory
    └── face_presence.csv # Output CSV (generated)
```

## Notes

- Model weights are automatically downloaded on first run
- Processing time depends on image size and number of images
- GPU acceleration is available if PyTorch/CUDA is configured
