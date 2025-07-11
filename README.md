# Benchmarking Cross-Domain Few-Shot Object Detection in Aerial Imagery

This repository contains the official implementation of our paper:\
📄 Benchmarking Cross-Domain Few-Shot Object Detection in Aerial Imagery

We implement LoRA (Low-Rank Adaptation) with DiffusionDet for cross-domain few-shot object detection, using the Hugging
Face Transformers framework. A custom library, `fsdetection`, adapts `HuggingFace`'s trainer and dataset components to
better support cross-domain object detection.

## 📂 Project Structure

```bash
├── configs/
│   ├── dataset/                # Template and metadata for your dataset
│   │   └── template.py         # Fill this to describe your dataset structure
│   ├── models/                 
│   │   └── template.json       # Example config for LoRA/DiffusionDet
│   └── dataset/
│       └── coco_format.py      # Script to convert COCO-format dataset to Hugging Face format
├── launch_experiments.py       # Multi-training launcher
├── run_object_detection.py     # Single training run
├── upload_to_hub.py            # Push datasets to the Hugging Face Hub
├── ...
```

## 📦 Dataset Preparation & Hugging Face Integration

To use your own COCO-format dataset:

- **Describe your dataset**\
  Fill out the file configs/dataset/template.py with information specific to your dataset (e.g., path, split keys, label
  mapping, etc.).

- **Convert your dataset (and optionally push to the Hub)**\
  You don't run the script coco_format.py directly. Instead, use the COCO class defined inside it to convert your
  dataset to Hugging Face format.

To simplify the process, you can just run:

```bash
python upload_to_hub.py --hf_repo your_hf_repo --dataset dataset_name
```

- **Using the dataset**\
  After pushing, your dataset is available to any script via datasets.load_dataset(...), or you can save and load it
  locally using Hugging Face’s dataset API.

## 🏁 Training
### 🔹**Single Training**

Use run_object_detection.py with arguments or a config file:

```bash
python run_object_detection.py ...
```

### 🔸**Multiple Trainings**

Use launch_experiments.py with a YAML file defining your sweep:

```bash
python launch_experiments.py --config-file configs/models/template.json ...
```

## 🌿 Branches

- `main`: Focused on cross-domain few-shot object detection.

- `classic-det`: Classic object detection use cases. (come soon)

\
📝 Note: Both branches use similar launch commands, but configuration files may differ slightly.

## 🔧 Requirements

Install dependencies and local packages:
```bash
pip install -r requirements.txt
```

## 📄 Citation

Coming soon

## 📨 Contact

For questions, feel free to open an issue or contact: [hich.tala.phd@gmail.com](mailto:hich.tala.phd@gmail.com)
