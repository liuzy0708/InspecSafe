"""
================================================================================
DMX API Batch Image Analysis Script (with Progress & ETA)
================================================================================
Description:
    Batch analyzes local images using specified multimodal model via DMX API,
    and saves results as .txt files named after each image.
================================================================================
"""

import base64
import json
import os
import time
import glob
from pathlib import Path
from datetime import datetime, timedelta

import requests


# ============================================================================
# Utility Functions
# ============================================================================

def encode_image(image_path):
    """Encode local image file to Base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_files(annotations_dir):
    """
    Recursively find all image files from Annotations directory
    (assumes images share the same name as annotations but with common image extensions)
    Note: Actual images may not be in Annotations directory, but in sibling directories like JPEGImages or images.
    This assumes images are in the same level as Annotations, or Annotations contains images (adjust based on your needs)
    """
    # Common image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(annotations_dir, '**', ext), recursive=True))
        # If images are in another directory (e.g., ../JPEGImages), add paths here
        # For example:
        # image_dir = os.path.join(annotations_dir, '..', 'JPEGImages')
        # image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    return sorted(image_paths)


# ============================================================================
# API Configuration
# ============================================================================

BASE_URL = "https://www.dmxapi.cn/"
API_ENDPOINT = BASE_URL + "v1/chat/completions"
API_KEY = "sk-***"

# Model name (used to create results folder)
MODEL_NAME = "claude-opus-4-5-20251101"

# Path Configuration
ANNOTATIONS_ROOT = "/path/to/your/DATA_PATH/test/Annotations"
MODEL_RESULTS_PATH = "/path/to/your/model_generate_results_dir/%s" % MODEL_NAME
os.makedirs(MODEL_RESULTS_PATH, exist_ok=True)
OUTPUT_DIR = os.path.join(MODEL_RESULTS_PATH, MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all image paths
image_paths = get_image_files(ANNOTATIONS_ROOT)
total_images = len(image_paths)

if total_images == 0:
    print(f"[!] No image files found in {ANNOTATIONS_ROOT} and its subdirectories.")
    exit(1)

print(f"[INFO] Found {total_images} images, results will be saved to: {OUTPUT_DIR}")

# Build prompt (unchanged)
prompt = (
    "Please analyze the input industrial scene image according to the following criteria:\n\n"
    "Task Objectives:\n"
    "1. Describe key visual elements in the image (e.g., personnel behavior, environmental conditions, object distribution, hazard sources, etc.).\n"
    "2. Assess the safety level of the scene into one of four categories:\n"
    "   - Level one: High-risk hazards present that could lead to severe incidents (e.g., open flame, smoke, personnel collapse, oil leakage, etc.);\n"
    "   - Level two: Moderate risks present that may cause safety issues or violations (e.g., using a mobile phone, oil accumulation, foreign objects, missing required PPE, etc.);\n"
    "   - Level three: Minor hazards present that require attention but are not urgent (e.g., water pooling, small foreign objects, missing face mask, etc.);\n"
    "   - no abnormalities observed: No obvious abnormalities or risk factors.\n\n"
    "Assessment Criteria (by industry):\n"
    "| Safety Level | Oil & Gas / Chemical | Coal Conveyor Gallery | Tunnel | Power | Metallurgy |\n"
    "|--------------|----------------------|------------------------|--------|-------|------------|\n"
    "| Level one      | Open flame, smoke, no hard hat, no gloves, no mask, smoking, personnel collapse, oil leakage | Open flame, smoke, no hard hat, smoking, personnel collapse | Open flame, smoke, non-motorized vehicles in fast lane, wood, metal, personnel collapse | Open flame, smoke, smoking, personnel collapse | Open flame, smoke, no hard hat, personnel collapse |\n"
    "| Level two      | Water pooling, using mobile phone | Using mobile phone, no gloves, no mask, foreign objects (plastic bags, bottles, foam, metal, paper) | Foam, plastic bags, plastic bottles, no hard hat, cabinet door open | Water pooling, oil accumulation, using mobile phone, foreign objects (plastic bags, bottles, foam, metal, paper), no hard hat, no gloves, cabinet door open | No gloves, no mask, smoking |\n"
    "| Level three      | Foreign objects | Water pooling | Water pooling, oil accumulation, using mobile phone, no gloves, no mask, smoking | No mask | Water pooling, oil accumulation, using mobile phone, foreign objects (plastic bags, bottles, foam, metal, paper) |\n\n"
    "Notes:\n"
    "- If the image cannot be clearly recognized, output 'Unrecognizable' and explain the reason in the [Image Description].\n"
    "- If the scene exhibits characteristics of multiple industries, prioritize the most relevant industry category.\n"
    "- If no risk factors are present, assign the safety level as 'no abnormalities observed'."
    "Output Format Requirements:\n"
    "Strictly follow the structure below (do not add extra content,only include Image Description and Safety Level):\n\n"
    "[Image Description]\n"
    "[Detailed description of the scene, human actions, environmental features, visible objects, etc.]\n\n"
    "[Safety Level]\n"
    "[Level one / Level two / Level three / no abnormalities observed]\n\n"
)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"{API_KEY}"
}

# ============================================================================
# Main Batch Processing Loop
# ============================================================================

processed = 0
total_time = 0.0
start_all = time.time()

for img_path in image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_file = os.path.join(OUTPUT_DIR, f"{img_name}.txt")

    # Skip already processed images
    if os.path.exists(output_file):
        print(f"[SKIP] Already exists: {img_name}")
        processed += 1
        continue

    try:
        # Encode image
        image_data = encode_image(img_path)
    except FileNotFoundError:
        print(f"[ERROR] Image not found: {img_path}")
        continue
    except Exception as e:
        print(f"[ERROR] Encoding failed {img_path}: {e}")
        continue

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ],
        "temperature": 0.1
    }

    start = time.time()
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60)
        elapsed = time.time() - start
        total_time += elapsed
        processed += 1

        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"[FAIL] {img_name} - {error_msg}")
            # Optional: save error info to file
            with open(output_file, 'w') as f:
                f.write(f"[API ERROR] {error_msg}\n")
            continue

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[OK] {img_name} ({elapsed:.2f}s)")
        else:
            error_detail = result.get("error", "Unknown error")
            print(f"[FAIL] {img_name} - No valid response: {error_detail}")
            with open(output_file, 'w') as f:
                f.write(f"[NO RESPONSE] {error_detail}\n")

    except Exception as e:
        elapsed = time.time() - start
        total_time += elapsed
        processed += 1
        print(f"[EXCEPTION] {img_name}: {e}")
        with open(output_file, 'w') as f:
            f.write(f"[EXCEPTION] {str(e)}\n")
        continue

    # Calculate ETA
    if processed > 0:
        avg_time = total_time / processed
        remaining = total_images - processed
        eta_seconds = avg_time * remaining
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        print(f"  -> Progress: {processed}/{total_images} | Avg time: {avg_time:.2f}s | ETA: {eta_str}")

# ============================================================================
# Final Statistics
# ============================================================================
total_elapsed = time.time() - start_all
print("\n" + "=" * 80)
print(f"Batch processing completed!")
print(f"Total images: {total_images}")
print(f"Processed/Skipped: {processed}")
print(f"Total time: {timedelta(seconds=int(total_elapsed))}")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)
