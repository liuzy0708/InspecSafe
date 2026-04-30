"""
Multimodal Robot Anomaly Detection Dataset Loader
Data Path Structure:
DATA_PATH/
├── train/
│   ├── Annotations/
│   │   ├── Normal_data/
│   │   │   └── {point_name}/
│   │   │       ├── {point_name}_visible_{timestamp}_frame_{frame_id}.jpg
│   │   │       ├── {point_name}_visible_{timestamp}_frame_{frame_id}.json
│   │   │       └── {point_name}_visible_{timestamp}_frame_{frame_id}.txt
│   │   └── Anomaly_data/
│   │       └── {anomaly_name}/
│   │           ├── {anomaly_name}.jpg
│   │           ├── {anomaly_name}.json
│   │           └── {anomaly_name}.txt
│   ├── Other_modalities/
│   │   └── {point_name}/
│   │       ├── {point_name}_visible_{timestamp}.mp4
│   │       ├── {point_name}_infrared_{timestamp}.mp4
│   │       ├── {point_name}_sensor_{timestamp}.txt
│   │       ├── {point_name}_point_cloud_{timestamp}.bag
│   │       └── {point_name}_audio_{timestamp}.wav
│   └── Parameters/
│       ├── Hardware/
│       └── Device_*.json
└── test/
    └── (same structure as train)
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Literal
from enum import Enum
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class DataSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class DataType(Enum):
    NORMAL = "Normal_data"
    ANOMALY = "Anomaly_data"


@dataclass
class ImageAnnotation:
    """Image annotation data structure."""
    image_path: str
    json_path: str
    txt_path: str
    label: int  # 0 for normal, 1 for anomaly
    data_type: DataType
    point_name: str
    frame_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalData:
    """Multimodal data for a collection point."""
    point_name: str
    rgb_video_path: Optional[str] = None
    infrared_video_path: Optional[str] = None
    sensor_data_path: Optional[str] = None
    point_cloud_path: Optional[str] = None
    audio_path: Optional[str] = None


class MultimodalRobotDataset(Dataset):
    """
    Multimodal Robot Anomaly Detection Dataset.

    Supports:
    - Loading normal and anomaly images with annotations
    - Loading multimodal data (RGB, infrared, sensor, point cloud, audio)
    - Loading device parameters
    """

    def __init__(
            self,
            root_path: str,
            split: DataSplit = DataSplit.TRAIN,
            data_type: Optional[DataType] = None,  # None means both
            transform=None,
            load_multimodal: bool = False,
            load_parameters: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            root_path: Root path to DATA_PATH directory
            split: DataSplit.TRAIN or DataSplit.TEST
            data_type: Filter by DataType, None means load all
            transform: Optional transform to apply to images
            load_multimodal: Whether to load multimodal data paths
            load_parameters: Whether to load device parameters
        """
        self.root_path = Path(root_path)
        self.split = split
        self.data_type = data_type
        self.transform = transform
        self.load_multimodal = load_multimodal
        self.load_parameters = load_parameters

        self.annotations: List[ImageAnnotation] = []
        self.multimodal_data: Dict[str, MultimodalData] = {}
        self.device_parameters: Dict[str, Any] = {}

        self._scan_dataset()

        if self.load_parameters:
            self._load_parameters()

    def _scan_dataset(self):
        """Scan and collect all annotation files."""
        split_path = self.root_path / self.split.value
        annotation_path = split_path / "Annotations"

        data_types = [self.data_type] if self.data_type else [DataType.NORMAL, DataType.ANOMALY]

        for dtype in data_types:
            dtype_path = annotation_path / dtype.value
            if not dtype_path.exists():
                continue

            for point_dir in dtype_path.iterdir():
                if not point_dir.is_dir():
                    continue

                point_name = point_dir.name

                # Find all image files
                for img_file in point_dir.glob("*.jpg"):
                    json_file = img_file.with_suffix(".json")
                    txt_file = img_file.with_suffix(".txt")

                    # Extract frame_id from filename
                    frame_id = self._extract_frame_id(img_file.name)

                    annotation = ImageAnnotation(
                        image_path=str(img_file),
                        json_path=str(json_file) if json_file.exists() else "",
                        txt_path=str(txt_file) if txt_file.exists() else "",
                        label=0 if dtype == DataType.NORMAL else 1,
                        data_type=dtype,
                        point_name=point_name,
                        frame_id=frame_id,
                    )
                    self.annotations.append(annotation)

        # Sort by point name for consistent ordering
        self.annotations.sort(key=lambda x: (x.data_type.value, x.point_name, x.frame_id or ""))

        # Load multimodal data if requested
        if self.load_multimodal:
            self._scan_multimodal_data()

    def _extract_frame_id(self, filename: str) -> Optional[str]:
        """Extract frame ID from filename."""
        # Pattern: *_frame_000001.jpg or frame_000001.jpg
        if "_frame_" in filename:
            parts = filename.replace(".jpg", "").split("_frame_")
            return parts[-1] if len(parts) > 1 else None
        elif filename.startswith("frame_"):
            return filename.replace(".jpg", "").replace("frame_", "")
        return None

    def _scan_multimodal_data(self):
        """Scan and collect multimodal data paths."""
        split_path = self.root_path / self.split.value
        multimodal_path = split_path / "Other_modalities"

        if not multimodal_path.exists():
            return

        for point_dir in multimodal_path.iterdir():
            if not point_dir.is_dir():
                continue

            point_name = point_dir.name
            mm_data = MultimodalData(point_name=point_name)

            for file in point_dir.iterdir():
                if "_visible_" in file.name and file.suffix == ".mp4":
                    mm_data.rgb_video_path = str(file)
                elif "_infrared_" in file.name and file.suffix == ".mp4":
                    mm_data.infrared_video_path = str(file)
                elif "_sensor_" in file.name and file.suffix == ".txt":
                    mm_data.sensor_data_path = str(file)
                elif "_point_cloud_" in file.name and file.suffix == ".bag":
                    mm_data.point_cloud_path = str(file)
                elif "_audio_" in file.name and file.suffix == ".wav":
                    mm_data.audio_path = str(file)

            self.multimodal_data[point_name] = mm_data

    def _load_parameters(self):
        """Load device parameters."""
        split_path = self.root_path / self.split.value
        params_path = split_path / "Parameters"

        if not params_path.exists():
            return

        # Load Device_*.json files
        for param_file in params_path.glob("*.json"):
            device_name = param_file.stem
            with open(param_file, 'r', encoding='utf-8') as f:
                self.device_parameters[device_name] = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - image: RGB image tensor (C, H, W)
            - label: 0 for normal, 1 for anomaly
            - json_data: annotation from json file
            - txt_data: semantic description from txt file
            - metadata: additional metadata
        """
        ann = self.annotations[idx]

        # Load image
        image = cv2.imread(ann.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Load json annotation
        json_data = None
        if ann.json_path and os.path.exists(ann.json_path):
            with open(ann.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

        # Load txt description
        txt_data = None
        if ann.txt_path and os.path.exists(ann.txt_path):
            with open(ann.txt_path, 'r', encoding='utf-8') as f:
                txt_data = f.read().strip()

        return {
            "image": image,
            "label": ann.label,
            "json_data": json_data,
            "txt_data": txt_data,
            "metadata": {
                "image_path": ann.image_path,
                "point_name": ann.point_name,
                "frame_id": ann.frame_id,
                "data_type": ann.data_type.value,
            }
        }

    def get_multimodal_data(self, point_name: str) -> Optional[MultimodalData]:
        """Get multimodal data for a specific point."""
        return self.multimodal_data.get(point_name)

    def get_parameter(self, device_name: str) -> Optional[Dict[str, Any]]:
        """Get device parameter by name."""
        return self.device_parameters.get(device_name)

    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        normal_count = sum(1 for a in self.annotations if a.data_type == DataType.NORMAL)
        anomaly_count = sum(1 for a in self.annotations if a.data_type == DataType.ANOMALY)

        return {
            "total": len(self.annotations),
            "normal": normal_count,
            "anomaly": anomaly_count,
            "points": len(set(a.point_name for a in self.annotations)),
            "multimodal_collections": len(self.multimodal_data),
            "device_parameters": len(self.device_parameters),
        }


def create_train_test_split(
        root_path: str,
        transform=None,
        load_multimodal: bool = False,
        load_parameters: bool = False,
) -> Tuple[MultimodalRobotDataset, MultimodalRobotDataset]:
    """
    Create train and test datasets.

    Args:
        root_path: Root path to DATA_PATH directory
        transform: Optional transform to apply to images
        load_multimodal: Whether to load multimodal data paths
        load_parameters: Whether to load device parameters

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = MultimodalRobotDataset(
        root_path=root_path,
        split=DataSplit.TRAIN,
        transform=transform,
        load_multimodal=load_multimodal,
        load_parameters=load_parameters,
    )

    test_dataset = MultimodalRobotDataset(
        root_path=root_path,
        split=DataSplit.TEST,
        transform=transform,
        load_multimodal=load_multimodal,
        load_parameters=load_parameters,
    )

    return train_dataset, test_dataset


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Robot Anomaly Dataset Loader")
    parser.add_argument("--root", type=str, default="/home/tc/trainData/multimodal_data_process/split1/DATA_PATH",
                        help="Root path to DATA_PATH directory")
    parser.add_argument("--split", type=str, choices=["train", "test", "all"], default="all",
                        help="Which split to load")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    if args.split == "all":
        train_ds, test_ds = create_train_test_split(
            root_path=args.root,
            load_multimodal=True,
            load_parameters=True,
        )

        print("=" * 50)
        print("TRAIN Dataset Statistics:")
        print("=" * 50)
        for k, v in train_ds.get_stats().items():
            print(f"  {k}: {v}")

        print("\n" + "=" * 50)
        print("TEST Dataset Statistics:")
        print("=" * 50)
        for k, v in test_ds.get_stats().items():
            print(f"  {k}: {v}")

    else:
        split = DataSplit.TRAIN if args.split == "train" else DataSplit.TEST
        dataset = MultimodalRobotDataset(
            root_path=args.root,
            split=split,
            load_multimodal=True,
            load_parameters=True,
        )

        print("=" * 50)
        print(f"{args.split.upper()} Dataset Statistics:")
        print("=" * 50)
        for k, v in dataset.get_stats().items():
            print(f"  {k}: {v}")

    # Test loading a sample
    if args.stats and args.split == "all":
        print("\n" + "=" * 50)
        print("Sample Data (first 3):")
        print("=" * 50)
        for i, sample in enumerate(train_ds):
            if i >= 3:
                break
            print(f"\nSample {i + 1}:")
            print(f"  Label: {sample['label']} ({'Normal' if sample['label'] == 0 else 'Anomaly'})")
            print(f"  Point: {sample['metadata']['point_name']}")
            print(f"  Frame: {sample['metadata']['frame_id']}")
            print(f"  Image shape: {sample['image'].shape if hasattr(sample['image'], 'shape') else 'N/A'}")
