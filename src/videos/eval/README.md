# Fire Detection Evaluation

This project is a tool for evaluating fire detection in images using computer vision models. The script processes images in folders, applies a fire detection model, and saves the results and evaluation metrics in organized folders.

## Requirements

- Python 3.x
- Necessary libraries (can be installed using `requirements.txt`):
- matplotlib
- opencv-python
- Pillow
- ultralytics
- torch
- tqdm

## Configuration

The configuration file `config/example.json` defines the necessary parameters to run the evaluation.

### Example Configuration
#### Yolo+CNN-LSTM
```json
{
    "main_path": "data",
    "output_path": "experiments",
    "model_type": "yolov8",
    "model_version": "pyronear2024",
    "lstm_resnet_model_type": "lstm_resnet",
    "lstm_resnet_model_version": "fc0",
    "ignition_time_path": "data/fire_truth.json",
    "video_folder": "20190529_94Fire_lp-s-mobo-c",
    "confidence_threshold": 0.05,
    "frames_back": 5
}
```
#### Yolo
```json
{
    "main_path": "data",
    "output_path": "experiments",
    "model_type": "yolov8",
    "model_version": "best",
    "lstm_resnet_model_type": "",
    "lstm_resnet_model_version": "",
    "ignition_time_path": "data/fire_truth.json",
    "video_folder": "",
    "confidence_threshold": 0.20,
    "frames_back": ""
}
```
### Parameter Descriptions

- `main_path`: Path to the main folder containing the image folders to be processed.

- `output_path`: Path where the evaluation results will be saved.

- `model_type`: Type of detection model to use.

- `model_version`: Version of the detection model to use.

- `lstm_resnet_model_type`: Type of the temporal detection model to use.

- `lstm_resnet_model_version`: Version of the temporal detection model to use.

- `ignition_time_path`: Path to the file containing the ignition times.

- `video_folder`: (Optional) Name of a specific video folder to process. If not specified, all folders in main_path will be processed.

- `confidence_threshold`: Confidence threshold to consider a detection as valid.

- `frames_back`: Number of frames brack in the temporal detection model.

## Execution
1. Ensure all necessary libraries are installed.
2. Modify the configuration file config/example.json with the appropriate parameters or introduce your configuration file.
3. Run the script eval.py with the name of your configuration as parameter:
```batch
python eval.py --config config/example.json
```
