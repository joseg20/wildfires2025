# Training Utilities for Images

This folder contains two scripts used to obtain datasets and train image-based smoke detection models. The code is designed around the [Ultralytics YOLO](https://docs.ultralytics.com) framework and integrates experiment tracking through [Weights & Biases](https://wandb.ai/).

## download.py

`download.py` automates downloading and extracting training datasets from Google Drive. Each dataset is identified by a numerical option and defined inside the script. The file is downloaded using `gdown`, extracted locally, and the original archive is removed.

Usage example:
```bash
python download.py 13
```
This command will download the datasets whose options are `1` and `3` to the `./data` directory.

## train.py

`train.py` provides a command-line interface for training a YOLO model. It initializes a Weights & Biases run and then calls `YOLO.train()` with the specified parameters.

Key arguments:
- `--model_weights`: path to pretrained weights (default: `yolov5s.pt`).
- `--data_config`: path to the dataset YAML configuration.
- `--epochs`: total training epochs (default: `100`).
- `--img_size`: input image size (default: `640`).
- `--batch_size`: training batch size (default: `16`).
- `--devices`: comma-separated list of GPU indices.
- `--project`: directory where the run results will be stored (default: `runs/train`).

A typical command looks like:
```bash
python train.py --data_config data/wildfire.yaml --epochs 200 --devices 0
```
After training, the best model weights are saved under `<project>/<run-name>/weights/best.pt`.

## Requirements

Dependencies are listed in `../../requirements.txt`. Install them via:
```bash
pip install -r requirements.txt
```