# UAV Human Detection with YOLO

Computer vision project for detecting people in UAV imagery. The project was developed as part of university project work.

The task was motivated by search-and-rescue scenarios where people need to be detected in large forest areas using images from unmanned aerial vehicles.

## Project Scope

The ML/CV implementation included:

- selecting a YOLO-based object detection approach;
- setting up a UAV human-detection dataset;
- training and evaluating the model;
- saving evaluation metrics;
- visualizing predictions on test images.

The original project presentation describes the system goal as a computer vision module for detecting people in UAV imagery in forest environments.

## Dataset

The experiment used the UAV-Lacmus dataset from Roboflow:

- dataset: UAV-Lacmus v7;
- source: https://universe.roboflow.com/crumbobly/uav-lacmus/dataset/7;
- license metadata: CC BY 4.0;
- annotation format: YOLOv8;
- class: `human`.

The full dataset is not included in this repository. To run training or evaluation, download the dataset from Roboflow and place it under `data/` with this structure:

```text
data/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

## Method

The saved experiment used a YOLO11m-based object detection workflow with PyTorch/CUDA and Ultralytics YOLO.

Model weights are not included in this repository. The original trained checkpoint was not preserved; the `yolo11m.pt` file in the archived working folder was a pretrained/base model, not the final trained model.

## Results

The preserved experiment results are stored in:

```text
results/
```

Preserved metrics from that experiment:

| Metric | Value |
| --- | ---: |
| mAP@0.5 | 0.7949 |
| mAP@0.5:0.95 | 0.3604 |
| Precision | 0.8007 |
| Recall | 0.7586 |

These metrics are archived experiment results. Because the trained checkpoint was not preserved, the exact run is not fully reproducible from this repository alone.

Input image examples are available in `assets/examples/`.

Selected annotated prediction examples are available in `results/examples/`.

## Project Structure

```text
.
  assets/examples/      # Curated input image examples
  configs/data.yaml     # Dataset configuration
  results/              # Preserved metrics and annotated examples
  scripts/train.py      # Training entry point
  scripts/evaluate.py   # Evaluation entry point
  scripts/infer_images.py
  requirements.txt
```

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Roboflow and place it under `data/` as described above.

## Training

Example command:

```bash
python scripts/train.py --data configs/data.yaml --weights yolo11m.pt --epochs 150 --device 0
```

Ultralytics may download the base model weights if they are not already available locally.

## Evaluation

Evaluation requires a trained checkpoint:

```bash
python scripts/evaluate.py --weights runs/detect/uav-human-detection-yolo11m/weights/best.pt --data configs/data.yaml --split test
```

## Inference

Run inference on images with a trained checkpoint:

```bash
python scripts/infer_images.py --weights runs/detect/uav-human-detection-yolo11m/weights/best.pt --source data/test/images --output results/predictions --conf 0.3
```

## Notes

- The full dataset is not stored in this repository.
- Model weights are not stored in this repository.
- The original trained checkpoint was not preserved.
- The saved metrics are included as archived experiment results.
- The original video artifact is not included because it is not part of the current project package.
- This repository does not claim production readiness.
