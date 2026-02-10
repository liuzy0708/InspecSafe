
# InspecSafe-V1: A Multimodal Benchmark for Safety Assessment in Real-World Industrial Inspection

<p align="center">
  <!-- Dataset -->
  <a href="https://huggingface.co/datasets/Tetrabot2026/InspecSafe-V1">
    <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface&logoColor=black" />
  </a>

  <!-- Paper -->
  <a href="https://arxiv.org/abs/2601.21173">
    <img alt="arXiv" src="https://img.shields.io/badge/Paper-arXiv%202601.21173-b31b1b?logo=arxiv&logoColor=white" />
  </a>

  <!-- Version -->
  <img alt="Version" src="https://img.shields.io/badge/Version-v1.0-blue" />

  <!-- Scale -->
  <img alt="Instances" src="https://img.shields.io/badge/Instances-5013-informational" />
  <img alt="Inspection Points" src="https://img.shields.io/badge/Inspection%20Points-2239-informational" />
  <img alt="Robots" src="https://img.shields.io/badge/Robots-41-informational" />
  <img alt="RGB Classes" src="https://img.shields.io/badge/RGB%20Classes-234-informational" />

  <!-- Modalities -->
  <img alt="Modalities" src="https://img.shields.io/badge/Modalities-RGB%20%7C%20Thermal%20%7C%20Audio%20%7C%20PointCloud%20%7C%20Sensors%20%7C%20Language-informational" />

  <!-- Tasks -->
  <img alt="Tasks" src="https://img.shields.io/badge/Tasks-Safety%20Assessment%20%7C%20Segmentation%20%7C%20VLM%20Reasoning-brightgreen" />

  <!-- Repo status 
  <a href="https://github.com/liuzy0708/InspecSafe/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/liuzy0708/InspecSafe?style=social" />
  </a>
  -->
  <!-- Citation -->
  <a href="#citation">
    <img alt="Cite" src="https://img.shields.io/badge/Cite-BibTeX-blue" />
  </a>

  <!-- License -->
  <a href="#license">
    <img alt="License" src="https://img.shields.io/badge/License-Research%20Use-lightgrey" />
  </a>
</p>

InspecSafe-V1 is a real-world multimodal benchmark designed for **safety assessment in industrial inspection scenarios**. The dataset is collected from routine inspection missions performed by deployed inspection robots across multiple industrial sites. It targets **vision–language safety reasoning** (scene understanding + safety-level judgement) under challenging real-world conditions (occlusion, glare/reflection, cluttered backgrounds, illumination variation, noise, etc.).

<p align="center">
  <img width="886" height="553" alt="截屏2026-02-10 22 38 33" src="https://github.com/user-attachments/assets/fe60deb8-da6a-4a97-9e1d-21fbc2d31ba5" />
</p>


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

<p align="center">
<img width="890" height="573" alt="截屏2026-02-10 22 40 23" src="https://github.com/user-attachments/assets/3622f4d4-00e0-410f-94e9-aa347143aaf8" />
</p>


---

## Robot Platforms

Two categories of inspection robots are involved:

- **Wheeled inspection robots**
- **Rail-mounted / suspended-rail inspection robots**

<p align="center">
<img width="911" height="313" alt="截屏2026-02-10 22 39 22" src="https://github.com/user-attachments/assets/750ad1e7-cf7a-40f7-b2df-aec8b8f5e816" />
</p>



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

<p align="center">
<img width="876" height="491" alt="截屏2026-02-10 22 41 34" src="https://github.com/user-attachments/assets/5cf2efa2-9be6-4f3b-8e87-ae090828b030" />
</p>


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

<p align="center">
<img width="585" height="508" alt="截屏2026-02-10 22 42 35" src="https://github.com/user-attachments/assets/e7f41180-edd3-4543-bbf2-8198bff0e731" />
</p>


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
This dataset is developed by researchers and collaborators from institutions including Department of Automation and the Institute for Embodied Intelligence and Robotics of Tsinghua University and TetraBOT Intelligence Co., Ltd. The public release is hosted on Hugging Face.
