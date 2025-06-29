# Scrapping The Web For Early Wildfire Detection

This repository accompanies the paper **"Scrapping The Web For Early Wildfire Detection: A New Annotated Dataset of Images and Videos of Smoke Plumes In-the-wild"**. It provides the code used to train and evaluate smoke detection models on still images and video sequences.

## Repository layout

- **src/images** – YOLO-based single‑frame detector. Contains training and evaluation scripts for the one‑frame approach.
- **src/videos** – Sequential detection code. Includes the tools to train and evaluate temporal CNN‑LSTM models.

Each folder has its own `README.md` with detailed usage instructions.

## Dataset

The smoke dataset used in the paper is available from [Google Drive](https://drive.google.com/drive/folders/1OiK_2iPGESRzLatNYQZFbrDatu0hZ4FE?usp=drive_link).