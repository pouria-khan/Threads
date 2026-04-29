# Text Extraction Pipeline (Stage 5) — PaddleOCR v4

## Purpose

This pipeline extracts all text blocks detected within social-media images using PaddleOCR v4. Raw output is one row per detected text region, with bounding-box coordinates and confidence scores. Output data feeds downstream derivation of text-image variables (`to_size`: text-image ratio; `to_centrality`: text centrality relative to image centre).

**Scale:** ~350,000 social-media images (Instagram, Threads); reproducible, resumable, fully logged.

---

## Inputs

### Required
- `--image_dir`: Directory (recursive) containing images in `.jpg`, `.jpeg`, `.png` format

### Optional
- `--output_dir`: Output CSV directory (default: `output/`)
- `--log_dir`: Log directory (default: `logs/`)
- `--docs_dir`: Documentation output directory (default: `docs/`)
- `--use_gpu` / `--no_gpu`: GPU acceleration toggle (default: `--use_gpu`)
- `--num_workers`: Number of parallel workers (default: 4 for GPU, CPU count − 1 for CPU)

---

## Outputs

### Main Output: `output/ocr_results.csv`

One row per detected text block per image. Columns:

| Column | Type | Description |
|---|---|---|
| `image_filename` | `str` | Filename only (not path) |
| `block_id` | `int` | 0-indexed position within image; 0 for no-text rows |
| `text` | `str` | Raw OCR text; null for images with no text |
| `confidence` | `float` | Detection confidence [0–1], rounded to 4 dp; null for no-text |
| `x_min`, `y_min` | `int` | Top-left corner of bounding box (pixels) |
| `x_max`, `y_max` | `int` | Bottom-right corner of bounding box (pixels) |
| `box_width` | `int` | Bounding box width in pixels |
| `box_height` | `int` | Bounding box height in pixels |
| `box_area_pct` | `float` | Percentage of image area occupied by text; rounded to 4 dp |
| `error_flag` | `int` | 0 = success; 1 = corrupted/unreadable image |

**Null handling:** Images with no detected text have one row: `block_id=0` with all other fields (except `image_filename`, `error_flag`) null (empty cell in CSV).

### Logs

Two timestamped log files per run:

- `logs/run_YYYYMMDD_HHMMSS.log` — all images (success and failure)
- `logs/errors_YYYYMMDD_HHMMSS.log` — errors only

Log format: `timestamp | LEVEL | image_filename | status | [key values]`

Progress logged every 1,000 images. Run summary block printed at completion.

### Model Documentation

`docs/model_documentation.docx` — automatically generated Word document with:
- Model identity and architecture
- Pretraining data
- Inference settings
- Processing statistics (from run summary)
- Full APA 7th citation

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke Test (5–10 images)

```bash
python detect_ocr.py --image_dir /path/to/images --use_gpu --num_workers 4
```

Verify:
- `output/ocr_results.csv` exists and has correct columns
- `logs/run_YYYYMMDD_HHMMSS.log` has one line per image + summary block
- `docs/model_documentation.docx` is generated

### 3. Full Pipeline (350K images)

```bash
python detect_ocr.py --image_dir /path/to/images --use_gpu --num_workers 4
```

If interrupted, restart with the same command — already-processed images are skipped (resumability).

### 4. CPU-Only Mode

```bash
python detect_ocr.py --image_dir /path/to/images --no_gpu --num_workers 8
```

---

## Dependencies

- **PaddleOCR v4**: ≥2.8.0 (text detection and recognition)
- **PaddlePaddle**: ≥2.6.0 (framework for PaddleOCR)
- **OpenCV**, **pandas**, **tqdm**, **numpy**, **Pillow**: standard image/data processing
- **python-docx**: .docx generation

All pinned in `requirements.txt`.

---

## Model: PaddleOCR v4

**Architecture:** Cascade of three neural networks:
1. **Detection**: ResNet50 + FPN + region-proposal heads (text localisation)
2. **Recognition**: CRNN with CTC loss (character-level prediction)
3. **Angle classifier** (optional; enabled here): handles rotated/curved text

**Pretraining data:** ICDAR 2013, 2015, 2019; COCO-Text, Total-Text, ArT; SynthText. Multilingual. Known biases: performs better on printed/clear text than handwriting; varies by language.

**Inference settings used:**
- `use_angle_cls=True` — enable angle classifier
- `lang="en"` — English OCR weights
- `use_gpu=True` (by default) — NVIDIA CUDA acceleration
- No confidence filtering — all detected text blocks extracted

**Known failure modes:**
- Very small text (<10 px) often missed
- Degraded/blurry text: noisy recognition
- Non-Latin scripts: may underperform

---

## Resumability

On startup, the script checks `output/ocr_results.csv` and skips images already processed. An interrupted 350K run can be restarted from its stopping point without reprocessing.

---

## Validation

Smoke-tested on 8 real social-media images. Manual verification of bounding-box accuracy, text extraction correctness, and proper handling of edge cases (no text, rotated text, overlapping text).

---

## Citation

If using this pipeline in a publication, cite PaddleOCR:

**APA 7th:**
> Du, Y., Li, C., Guo, R., Yin, X., Liu, W., Zhou, J., ... & Wang, Y. (2020). Towards Accurate Scene Text Recognition with Semantic Reasoning Networks. *arXiv preprint arXiv:2009.09941*.

---

## Contact

Prashant Sharma, Ivey Business School  
Reza Khansari

---

## Version

Stage 5 — PaddleOCR v4  
Last updated: 2026-04-29
