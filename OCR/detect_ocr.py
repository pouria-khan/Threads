#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCR not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

from utils import (
    get_image_files,
    load_image,
    validate_directory,
    extract_bbox_from_quad,
    generate_model_documentation,
)

_ocr_engine = None


def setup_logging(log_dir: str) -> Tuple[str, str]:
    """Setup logging to two files (run log and error log) with timestamps."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validate_directory(log_dir)

    run_log = os.path.join(log_dir, f"run_{timestamp}.log")
    error_log = os.path.join(log_dir, f"errors_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [
        logging.FileHandler(run_log),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    return run_log, error_log


def get_processed_images(csv_path: str) -> set:
    """Read existing CSV and return set of already-processed image filenames."""
    if not os.path.exists(csv_path):
        return set()

    try:
        df = pd.read_csv(csv_path)
        return set(df["image_filename"].unique())
    except Exception as e:
        logging.warning(f"Could not read existing CSV: {e}")
        return set()


def process_image_gpu(
    image_path: str, error_log_path: str
) -> Tuple[str, List[Dict], Optional[str]]:
    """
    Process a single image using GPU-shared OCR engine.
    Returns (filename, list of row dicts, error_msg or None)
    """
    filename = os.path.basename(image_path)
    start_time = time.time()

    try:
        img, error_flag = load_image(image_path)
        if error_flag:
            err_msg = "OSError: cannot identify image file"
            with open(error_log_path, "a") as ef:
                ef.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | {filename} | corrupted | {err_msg}\n"
                )
            return filename, [], err_msg

        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width

        result = _ocr_engine.ocr(image_path)

        if not result or not result[0] or not result[0].get("rec_texts"):
            row = {
                "image_filename": filename,
                "block_id": 0,
                "text": None,
                "confidence": None,
                "x_min": None,
                "y_min": None,
                "x_max": None,
                "y_max": None,
                "box_width": None,
                "box_height": None,
                "box_area_pct": None,
                "error_flag": 0,
            }
            duration = time.time() - start_time
            logging.info(f"{filename} | processed | blocks=0 | duration={duration:.2f}s")
            return filename, [row], None

        ocr_result = result[0]
        texts = ocr_result.get("rec_texts", [])
        scores = ocr_result.get("rec_scores", [])
        polys = ocr_result.get("rec_polys", [])

        rows = []
        for block_id, (text, confidence, poly) in enumerate(zip(texts, scores, polys)):
            x_min, y_min, x_max, y_max = extract_bbox_from_quad(poly.tolist())
            box_width = x_max - x_min
            box_height = y_max - y_min
            box_area = box_width * box_height
            box_area_pct = (box_area / img_area * 100) if img_area > 0 else 0.0

            row = {
                "image_filename": filename,
                "block_id": block_id,
                "text": text,
                "confidence": round(float(confidence), 4),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "box_width": box_width,
                "box_height": box_height,
                "box_area_pct": round(box_area_pct, 4),
                "error_flag": 0,
            }
            rows.append(row)

        duration = time.time() - start_time
        logging.info(f"{filename} | processed | blocks={len(rows)} | duration={duration:.2f}s")
        return filename, rows, None

    except Exception as e:
        err_msg = str(e)
        with open(error_log_path, "a") as ef:
            ef.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | {filename} | failed | {err_msg}\n"
            )
        logging.error(f"{filename} | error | {err_msg}")
        return filename, [], err_msg


def _init_worker_cpu(use_gpu: bool):
    """Initializer for CPU multiprocessing pool."""
    global _ocr_engine
    _ocr_engine = PaddleOCR(lang="en")


def process_image_cpu(args) -> Tuple[str, List[Dict], Optional[str]]:
    """Process image in CPU worker process."""
    image_path, error_log_path = args
    filename = os.path.basename(image_path)
    start_time = time.time()

    try:
        img, error_flag = load_image(image_path)
        if error_flag:
            err_msg = "OSError: cannot identify image file"
            with open(error_log_path, "a") as ef:
                ef.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | {filename} | corrupted | {err_msg}\n"
                )
            return filename, [], err_msg

        img_height, img_width = img.shape[:2]
        img_area = img_height * img_width

        result = _ocr_engine.ocr(image_path)

        if not result or not result[0] or not result[0].get("rec_texts"):
            row = {
                "image_filename": filename,
                "block_id": 0,
                "text": None,
                "confidence": None,
                "x_min": None,
                "y_min": None,
                "x_max": None,
                "y_max": None,
                "box_width": None,
                "box_height": None,
                "box_area_pct": None,
                "error_flag": 0,
            }
            return filename, [row], None

        ocr_result = result[0]
        texts = ocr_result.get("rec_texts", [])
        scores = ocr_result.get("rec_scores", [])
        polys = ocr_result.get("rec_polys", [])

        rows = []
        for block_id, (text, confidence, poly) in enumerate(zip(texts, scores, polys)):
            x_min, y_min, x_max, y_max = extract_bbox_from_quad(poly.tolist())
            box_width = x_max - x_min
            box_height = y_max - y_min
            box_area = box_width * box_height
            box_area_pct = (box_area / img_area * 100) if img_area > 0 else 0.0

            row = {
                "image_filename": filename,
                "block_id": block_id,
                "text": text,
                "confidence": round(float(confidence), 4),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "box_width": box_width,
                "box_height": box_height,
                "box_area_pct": round(box_area_pct, 4),
                "error_flag": 0,
            }
            rows.append(row)

        return filename, rows, None

    except Exception as e:
        err_msg = str(e)
        with open(error_log_path, "a") as ef:
            ef.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ERROR | {filename} | failed | {err_msg}\n"
            )
        return filename, [], err_msg


def main():
    parser = argparse.ArgumentParser(
        description="Extract text blocks from images using PaddleOCR v4"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--docs_dir", type=str, default="docs", help="Documentation directory")
    parser.add_argument(
        "--use_gpu", action="store_true", default=True, help="Use GPU for inference (default: True)"
    )
    parser.add_argument(
        "--no_gpu", dest="use_gpu", action="store_false", help="Force CPU-only inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    validate_directory(args.output_dir)
    validate_directory(args.log_dir)
    validate_directory(args.docs_dir)

    run_log_path, error_log_path = setup_logging(args.log_dir)
    output_csv = os.path.join(args.output_dir, "ocr_results.csv")

    start_time_global = datetime.now()
    logging.info(f"Starting OCR pipeline (GPU: {args.use_gpu})")
    logging.info(f"Image directory: {args.image_dir}")

    all_images = get_image_files(args.image_dir)
    processed_set = get_processed_images(output_csv)
    todo_images = [img for img in all_images if os.path.basename(img) not in processed_set]

    logging.info(f"Total images found: {len(all_images)}")
    logging.info(f"Already processed: {len(processed_set)}")
    logging.info(f"To process: {len(todo_images)}")

    if not todo_images:
        logging.info("All images already processed. Exiting.")
        return

    if args.num_workers is None:
        args.num_workers = 4 if args.use_gpu else max(1, cpu_count() - 1)

    total_processed = 0
    total_errors = 0
    all_rows = []

    if args.use_gpu:
        global _ocr_engine
        logging.info(f"Initializing PaddleOCR (GPU) with {args.num_workers} threads...")
        _ocr_engine = PaddleOCR(lang="en")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_image_gpu, img_path, error_log_path): img_path
                for img_path in todo_images
            }

            with tqdm(total=len(todo_images), unit="image", desc="Processing") as pbar:
                for i, future in enumerate(as_completed(futures), 1):
                    filename, rows, error_msg = future.result()

                    if rows:
                        all_rows.extend(rows)
                        total_processed += 1

                    if error_msg:
                        total_errors += 1

                    pbar.update(1)

                    if i % 1000 == 0:
                        logging.info(
                            f"PROGRESS | {i}/{len(todo_images)} images ({i/len(todo_images)*100:.1f}%)"
                        )

    else:
        logging.info(f"Initializing PaddleOCR (CPU) with {args.num_workers} processes...")

        with Pool(
            processes=args.num_workers, initializer=_init_worker_cpu, initargs=(False,)
        ) as pool:
            image_args = [(img, error_log_path) for img in todo_images]

            with tqdm(total=len(todo_images), unit="image", desc="Processing") as pbar:
                for i, (filename, rows, error_msg) in enumerate(
                    pool.imap_unordered(process_image_cpu, image_args), 1
                ):
                    if rows:
                        all_rows.extend(rows)
                        total_processed += 1

                    if error_msg:
                        total_errors += 1

                    pbar.update(1)

                    if i % 1000 == 0:
                        logging.info(
                            f"PROGRESS | {i}/{len(todo_images)} images ({i/len(todo_images)*100:.1f}%)"
                        )

    if all_rows:
        df = pd.DataFrame(all_rows)

        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            df = pd.concat([existing_df, df], ignore_index=True)

        column_order = [
            "image_filename",
            "block_id",
            "text",
            "confidence",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "box_width",
            "box_height",
            "box_area_pct",
            "error_flag",
        ]
        df = df[column_order]
        df.to_csv(output_csv, index=False)
        logging.info(f"Results written to {output_csv}")

    end_time_global = datetime.now()
    duration = (end_time_global - start_time_global).total_seconds()
    avg_time = duration / len(todo_images) if len(todo_images) > 0 else 0

    summary = f"""
========== RUN SUMMARY ==========
Started:        {start_time_global.strftime('%Y-%m-%d %H:%M:%S')}
Finished:       {end_time_global.strftime('%Y-%m-%d %H:%M:%S')}
Total images:   {len(all_images)}
Processed:      {total_processed}
Skipped:        {len(processed_set)}
Errors:         {total_errors}  ({total_errors / len(all_images) * 100:.2f}%)
Avg time/image: {avg_time:.3f}s
Model:          PaddleOCR v4 (paddleocr>=2.8.0)
Output file:    {output_csv}
=================================
"""

    logging.info(summary)

    doc_path = os.path.join(args.docs_dir, "model_documentation.docx")
    generate_model_documentation(doc_path, len(all_images), total_processed, total_errors, avg_time)

    logging.info("OCR pipeline complete.")


if __name__ == "__main__":
    main()
