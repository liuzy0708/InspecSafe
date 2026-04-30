import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

MODEL_NAME = "grok-4.1-fast"
MODEL_RESULTS_PATH = "/path/to/your/model_generate_results_dir/%s/" % MODEL_NAME
GT_ROOT_ANOMALY = "/path/to/your/DATA_PATH/test/Annotations/Anomaly_data"
GT_ROOT_NORMAL = "/path/to/your/DATA_PATH/test/Annotations/Normal_data"

# Define class order
classes = ["level one", "level two", "level three", "no abnormalities observed", "unrecognizable"]
tick_label_classes = ["level Ⅰ", "level Ⅱ", "level Ⅲ", "level Ⅳ", "unrecognizable"]

# Ground truth label mapping
label_map = {
    "observed": "no abnormalities observed",
    "one": "level one",
    "two": "level two",
    "ii": "level two",
    "2": "level two",
    "three": "level three",
    "unrecognizable": "unrecognizable",
}


def extract_prediction(file_path):
    """Extract the last word from prediction file and map to standard class"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        return label_map["unrecognizable"]
    last_line = lines[-1].strip()
    if '(' in last_line:
        last_line = last_line.split('(')[0]

    words = last_line.split()
    if not words:
        return label_map["unrecognizable"]

    last_word = words[-1]
    # Remove possible punctuation (e.g., period)
    last_word = last_word.rstrip('.').strip().lower().replace('level]', '').replace(']', '')
    if last_word in label_map:
        return label_map[last_word]
    else:
        print(f"Warning: Unknown prediction label keyword: '{last_word}' in {file_path}")
        return label_map["unrecognizable"]


def extract_ground_truth(file_path):
    """Extract the last word from ground truth file and map to standard class"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        return None
    last_line = lines[-1].strip()
    words = last_line.split()
    if not words:
        return None
    last_word = words[-1]
    # Remove possible punctuation (e.g., period)
    last_word = last_word.rstrip('.').strip().lower()
    if last_word in label_map:
        return label_map[last_word]
    else:
        print(f"Warning: Unknown ground truth label keyword: '{last_word}' in {file_path}")
        return None


def collect_files(root_dir):
    """Recursively collect all .txt files in directory, return {filename: full_path} dict"""
    file_dict = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.txt'):
                file_dict[f] = os.path.join(dirpath, f)
    return file_dict


def main():
    # Collect prediction and ground truth files
    pred_files = collect_files(MODEL_RESULTS_PATH)
    gt_files1 = collect_files(GT_ROOT_ANOMALY)
    gt_files2 = collect_files(GT_ROOT_NORMAL)
    gt_files = {**gt_files1, **gt_files2}

    # Match filenames
    common_files = set(pred_files.keys()) & set(gt_files.keys())
    print(f"Found {len(common_files)} matching samples")

    # Initialize confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    class_to_index = {cls: i for i, cls in enumerate(classes)}

    count_valid = 0
    for fname in common_files:
        pred_path = pred_files[fname]
        gt_path = gt_files[fname]

        pred = extract_prediction(pred_path)
        gt = extract_ground_truth(gt_path)

        if pred is None or gt is None:
            continue

        if pred not in class_to_index or gt not in class_to_index:
            print(f"Skip invalid class: pred={pred}, gt={gt}")
            continue

        i = class_to_index[gt]  # Ground truth -> row
        j = class_to_index[pred]  # Prediction -> column
        cm[i, j] += 1
        count_valid += 1

    print(f"Valid samples: {count_valid}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_label_classes,
                yticklabels=tick_label_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(MODEL_NAME)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()