# InspecSafe-V1  
**A Multimodal Benchmark for Safety Assessment in Real-World Industrial Inspection**

InspecSafe-V1 is a real-world multimodal benchmark designed for **safety assessment in industrial inspection scenarios**. The dataset is collected from routine inspection missions performed by deployed inspection robots across multiple industrial sites. It targets **vision–language safety reasoning** (scene understanding + safety-level judgement) under challenging real-world conditions (occlusion, glare/reflection, cluttered backgrounds, illumination variation, noise, etc.).

---

## Key Highlights

- Real-world industrial inspection data with safety-oriented annotations.
- Covers **5 industrial scenario categories** and **2 inspection-robot platforms** (wheeled and rail-mounted).
- Dataset scale:
  - **41** inspection robots  
  - **2,239** inspection points (sites)  
  - **5,013** inspection instances  
  - **234** RGB object classes (long-tailed)
- Per instance, the dataset provides at least:
  - **RGB keyframe** (from RGB video)
  - **Pixel-level polygon instance segmentation** (JSON)
  - **Textual scene description** (TXT)
  - **Safety level label** (**Level I–IV**, with **Level IV = no safety factor**)
- Multimodal sensing (depending on platform/site availability): **RGB**, **thermal/infrared**, **audio**, **point clouds**, **gas**, **temperature**, **humidity**.

---

## Covered Scenarios

InspecSafe-V1 includes five representative industrial environments:

1. Tunnel  
2. Power facilities  
3. Sintering / metallurgical equipment areas  
4. Oil & gas / petrochemical plants  
5. Coal conveyor trestles / coal transfer areas

---

## Robot Platforms

Two categories of inspection robots are involved:

- **Wheeled inspection robots**
- **Rail-mounted / suspended-rail inspection robots**

---

## Modalities and Typical File Formats

Each inspection point is associated with synchronized multimodal recordings (availability may vary by robot and site). Typical formats described in the accompanying paper include:

- **RGB video**: `.mp4` (short stop at each inspection point, typically ~10–15 s)
- **Thermal/IR video**: `.mp4`
- **Audio**: `.wav` (e.g., dual-channel; typical clip duration ~10–15 s)
- **Point clouds**: `.bag` (ROS bag; short capture window, e.g., a few seconds)
- **Environmental sensors (gas, temperature, humidity, etc.)**: `.txt`

---

## Annotation Types

### 1) Vision: Pixel-Level Polygon Instance Segmentation
For each RGB keyframe, InspecSafe-V1 provides **polygon-based instance segmentation** annotations in **JSON** format.

### 2) Language: Scene Description + Safety Semantics
Each instance includes a **text file** containing:
- a **scene description** (summarizing the visual context and salient events/objects), and
- a **safety level** label (**Level I–IV**).

> **Labeling rule:** if multiple hazards appear in one image, the final safety level is determined by the **most severe** hazard. If **no safety factor** is present, the label is **Level IV**.

### 3) Quality Control (QC)
The dataset construction process includes **multi-round independent verification** for both:
- pixel-level visual annotations, and
- text-level semantic annotations (descriptions + safety labels).

---

## Safety Levels (Level I–IV)

InspecSafe-V1 defines four discrete safety levels:

- **Level I**: highest risk  
- **Level II**: medium risk  
- **Level III**: lower risk  
- **Level IV**: **no safety factor / normal**

**Note:** the detailed criteria can be **scenario-dependent** (e.g., what constitutes Level I in oil & gas may differ from tunnel). Please refer to the dataset paper for the scenario-wise criteria table.

---

## Dataset Organization

The dataset is organized around **inspection instances** and is designed to support efficient access to (i) RGB keyframes and annotations, (ii) synchronized multimodal recordings, and (iii) auxiliary parameter files.

A high-level structure described in the paper includes:

- `Annotations/`
  - RGB keyframes and their labels
  - commonly separated into *normal* and *anomaly* subsets
  - per keyframe typically includes:
    - image file (`.jpg`/`.png`)
    - polygon annotation (`.json`)
    - language + safety label (`.txt`)
- `Other modalities/`
  - multimodal recordings aligned to inspection points/instances
- `Parameters/`
  - auxiliary parameter files for parsing/alignment (e.g., calibration/extrinsics, sensor configs, etc.)

> The release may also include **index/metadata files** to facilitate cross-modal alignment using identifiers and timestamps.

---

## Benchmark: VLM-Based Safety Assessment

InspecSafe-V1 is designed for evaluating **vision–language models (VLMs)** on industrial safety assessment. A typical benchmark setting includes:

- **Input**: RGB keyframe (+ a standardized prompt template)
- **Outputs**:
  1) a generated **scene description**, and  
  2) a predicted **safety level** (Level I–IV)

### Train/Test Split (as reported)

- **Train**: 3,763 frames (Normal: 3,014; Abnormal: 749)  
- **Test**: 1,250 frames (Normal: 999; Abnormal: 251)

To mitigate information leakage from highly similar adjacent frames within the same inspection point, the split is built with **uniform intra-point sampling** from RGB videos.

### Evaluation Metrics

- **Accuracy (Acc)**: safety-level classification accuracy  
- **Semantic Similarity (SemSim)**: cosine similarity between embeddings of
  - generated description vs. human description,  
  using a **fixed text encoder** (the paper reports using **BGE-M3**).

---

## Recommended Research Tasks

InspecSafe-V1 can support, but is not limited to:

- Safety-level classification (Level I–IV)
- Vision–language safety reasoning (joint description generation + risk judgement)
- Open-vocabulary detection / segmentation under industrial conditions
- Multimodal fusion (RGB + thermal + audio + point clouds + sensors)
- Cross-scenario generalization and robustness evaluation
- Long-tailed recognition in industrial object categories

---

## Data Access

The dataset is publicly available on **:contentReference[oaicite:0]{index=0}**:

- Dataset page: <https://huggingface.co/datasets/Tetrabot2026/InspecSafe-V1>

The release provides raw multimodal data, pixel-level annotations, scene descriptions, safety labels, and supporting resources (e.g., metadata/index files and parameter files).

## Data Access

The dataset is publicly available on **Hugging Face**:

- Dataset page: <https://huggingface.co/datasets/Tetrabot2026/InspecSafe-V1>

The release provides raw multimodal data, pixel-level annotations, scene descriptions, safety labels, and supporting resources (e.g., metadata/index files and parameter files).

---

## Usage Notes

- The dataset is intended for **research use**.
- Redistribution, modification, and derivative works are permitted under the dataset’s stated usage terms, with **proper citation required**.
- Privacy- or security-sensitive content has been **anonymized or removed** before release.

---

## Citation

If you use InspecSafe-V1 in your research, please cite the accompanying paper and the dataset page.

```bibtex
@misc{InspecSafe-V1,
      title={InspecSafe-V1: A Multimodal Benchmark for Safety Assessment in Industrial Inspection Scenarios}, 
      author={Zeyi Liu and Shuang Liu and Jihai Min and Zhaoheng Zhang and Jun Cen and Pengyu Han and Songqiao Hu and Zihan Meng and Xiao He and Donghua Zhou},
      year={2026},
      eprint={2601.21173},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.21173}, 
}
```

### Contact
For questions, issue reports, or collaboration requests, please use the dataset page’s contact channels and/or open an issue with:
- sample/instance identifier
- inspection point identifier
- timestamps (if relevant)
- modality file paths
- reproduction steps

### Acknowledgement
This dataset is developed by researchers and collaborators from institutions including Tsinghua University and TetroBot. The public release is hosted on Hugging Face.
