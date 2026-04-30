#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark Evaluation Script for Model Text Similarity
=========================================================
Compares generated results with reference texts using text embeddings.
"""

import numpy as np
import requests
import subprocess
import os
from pathlib import Path
from typing import List

MODEL_NAME = "grok-4.1-fast"
MODEL_RESULTS_PATH = "/path/to/your/model_generate_results_dir/%s/" % MODEL_NAME
TEST_DATA_PATH = "/path/to/your/DATA_PATH/test/"


class TextSimilarityCalculator:
    def __init__(self, model_name="bge-m3", ollama_host="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                return None

            payload = {"model": self.model_name, "prompt": text, "stream": False}
            response = requests.post(f"{self.ollama_host}/api/embeddings", json=payload, timeout=30)

            if response.status_code == 200:
                return response.json().get("embedding", [])
            return None
        except:
            return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0

        vec1, vec2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        embedding1, embedding2 = self.get_embedding(text1), self.get_embedding(text2)
        if embedding1 is None or embedding2 is None:
            return 0.0
        return float(self.cosine_similarity(embedding1, embedding2))

    def check_ollama_installation(self):
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                return self.model_name in result.stdout
            return False
        except:
            return False


def find_matching_txt_files(ref_dir, test_dir):
    matches = []

    ref_txt_files = list(Path(ref_dir).glob("*.txt"))

    for txt_path in Path(test_dir).rglob("*.txt"):
        txt_name = txt_path.name

        matching_ref = [ref for ref in ref_txt_files if ref.name == txt_name]

        if matching_ref:
            for ref_file in matching_ref:
                matches.append((ref_file, txt_path))

    return matches


def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ""


def main():
    matches = find_matching_txt_files(MODEL_RESULTS_PATH, TEST_DATA_PATH)

    if not matches:
        print("No matching txt files found")
        return

    print(f"Found {len(matches)} matching txt file pairs")
    print("-" * 50)

    calculator = TextSimilarityCalculator()

    if not calculator.check_ollama_installation():
        print("Ollama environment check failed")
        return

    similarities = []

    for i, (ref_path, test_path) in enumerate(matches, 1):
        ref_content = read_file_content(ref_path)
        test_content = read_file_content(test_path)

        if not ref_content or not test_content:
            print(f"File {ref_path.name}: Skipped (empty content)")
            continue

        similarity = calculator.calculate_similarity(ref_content, test_content)
        similarities.append(similarity)

        print(f"Pair {i}: {ref_path.name}")
        print(f"  Reference file: {ref_path}")
        print(f"  Target file: {test_path}")
        print(f"  Similarity: {similarity:.4f}")
        print("-" * 30)

    if similarities:
        avg_similarity = np.mean(similarities)
        print("=" * 50)
        print(f"Total file pairs: {len(similarities)}")
        print(f"Average similarity: {avg_similarity:.4f}")
    else:
        print("No valid file pairs for similarity calculation")


if __name__ == "__main__":
    main()
