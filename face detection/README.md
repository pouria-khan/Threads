# RetinaFace Face Detection Pipeline

A modular Python pipeline for detecting human faces in images using the RetinaFace model (Deng et al., 2020). Designed for social media post engagement research.

## Overview

This pipeline processes images in a directory and outputs:
- **face_presence**: Binary indicator (1 = face present, 0 = no face)
- **face_count**: Number of faces detected per image
- **error_flag**: Processing status (1 = failed, 0 = success)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python detect_faces.py --image_dir /path/to/images
```

### Optional Arguments
```bash
python detect_faces.py --image_dir /path/to/images --output_csv custom_output.csv
```

## Output

Results are saved to `output/face_presence.csv` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| image_filename | str | Name of the image file |
| face_presence | int | 1 if face detected, 0 otherwise |
| face_count | int | Number of faces detected |
| error_flag | int | 1 if processing failed, 0 if successful |

## Example Output

```
image_filename,face_presence,face_count,error_flag
photo_1.jpg,1,2,0
photo_2.jpg,0,0,0
photo_3.jpg,1,1,0
```

## Features

- ✅ Binary face detection for each image
- ✅ Face counting (number of faces per image)
- ✅ Robust error handling for corrupted/unreadable images
- ✅ Batch processing with progress bar (tqdm)
- ✅ Reproducible results with random seed control
- ✅ Well-commented code for academic researchers
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
