import os
import argparse
import json
from datetime import datetime
from models.model_loader import load_model
from utils.ignition_time import load_ignition_times
from utils.file_processor import process_files_in_folder
from utils.analysis import calculate_detection_metrics, save_metrics_as_txt
import statistics

CONFIG_PATH = 'config'

def save_config_to_file(config, output_folder):
    """
    Save the used configuration dict as JSON in the output folder.
    """
    config_file_path = os.path.join(output_folder, 'config_used.json')
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_file_path}")

def save_predictions_to_json(all_detection_data, output_file):
    """
    Write frame‐by‐frame predictions for each folder into a JSON file.
    """
    data_to_save = []
    for detection in all_detection_data:
        folder = detection['folder_name']
        predictions = detection["frame_predictions"]
        data_to_save.append([folder] + predictions)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f)
    print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Wildfire detection evaluation")
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(CONFIG_PATH, 'example.json'),
        help='Path to the configuration file'
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    main_path               = config['main_path']
    output_path             = config['output_path']
    yolo_model_type         = config['model_type']
    yolo_model_version      = config['model_version']
    lstm_resnet_model_type  = config['lstm_resnet_model_type']
    lstm_resnet_model_version = config['lstm_resnet_model_version']
    ignition_time_path      = config['ignition_time_path']
    video_folder            = config['video_folder']
    confidence_threshold    = config['confidence_threshold']
    frames_back             = config['frames_back']

    yolo_model = load_model(yolo_model_type, yolo_model_version)
    if lstm_resnet_model_type:
        lstm_resnet_model = load_model(lstm_resnet_model_type, lstm_resnet_model_version)
    else:
        lstm_resnet_model = None

    ignition_times = load_ignition_times(ignition_time_path)

    os.makedirs(output_path, exist_ok=True)

    evaluation_id     = f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    evaluation_folder = os.path.join(output_path, evaluation_id)
    os.makedirs(evaluation_folder)

    save_config_to_file(config, evaluation_folder)

    all_detection_data = []
    all_metrics        = []

    print(f"Processing files in {main_path}")
    print(f"Saving results to {evaluation_folder}")
    print(f"YOLO model: {yolo_model_type} (version {yolo_model_version})")
    print(f"LSTM-ResNet model: {lstm_resnet_model_type} (version {lstm_resnet_model_version})")
    print(f"Video folder setting: {video_folder!r}")

    if video_folder:
        folder_path   = os.path.join(main_path, video_folder)
        output_folder = os.path.join(evaluation_folder, video_folder)
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing folder: {folder_path}")
        detection_data = process_files_in_folder(
            folder_path, output_folder,
            yolo_model, lstm_resnet_model,
            ignition_times, confidence_threshold, frames_back
        )
        metric = calculate_detection_metrics(detection_data)
        all_metrics.append(metric)
        all_detection_data.append(detection_data)

    else:
        for root, dirs, files in os.walk(main_path):
            for dir_name in dirs:
                if "labels" in dir_name.lower():
                    print(f"Ignoring folder: {dir_name}")
                    continue

                folder_path   = os.path.join(root, dir_name)
                output_folder = os.path.join(evaluation_folder, dir_name)
                os.makedirs(output_folder, exist_ok=True)

                print(f"Processing folder: {folder_path}")
                detection_data = process_files_in_folder(
                    folder_path, output_folder,
                    yolo_model, lstm_resnet_model,
                    ignition_times, confidence_threshold, frames_back
                )
                if detection_data is not None:
                    metric = calculate_detection_metrics(detection_data)
                    all_metrics.append(metric)
                    all_detection_data.append(detection_data)

    # Save all per‐video predictions
    output_json_path = os.path.join(evaluation_folder, 'state_all_videos.json')
    save_predictions_to_json(all_detection_data, output_json_path)

    # Compute and save global precision/recall/F1 if available
    if all_metrics:
        valid_metrics = [m for m in all_metrics if m["precision"] is not None]
        if valid_metrics:
            avg_precision = statistics.mean(m["precision"] for m in valid_metrics)
            avg_recall    = statistics.mean(m["recall"]    for m in valid_metrics)
            avg_f1        = statistics.mean(m["f1_score"]  for m in valid_metrics)

            global_metrics = {
                "precision": avg_precision,
                "recall":    avg_recall,
                "f1_score":  avg_f1
            }
            global_metrics_file = os.path.join(evaluation_folder, 'global_metrics.txt')
            save_metrics_as_txt(global_metrics, evaluation_folder, filename='global_metrics.txt')
            print(f"Global metrics saved to {global_metrics_file}")
        else:
            print("No files with sufficient metrics to calculate averages.")

    # Compute and save delay statistics
    delays          = []
    delay_file_path = os.path.join(evaluation_folder, 'delay_metrics.txt')
    with open(delay_file_path, 'w', encoding='utf-8') as delay_file:
        delay_file.write("Delays by folder:\n")
        delay_file.write("-----------------\n")
        for detection_data in all_detection_data:
            folder_name = detection_data['folder_name']
            delay       = detection_data.get('detection_delay', None)
            if isinstance(delay, (int, float)):
                delays.append(delay)
                delay_str = f"{delay:.2f} seconds"
            else:
                delay_str = "N/A"
            delay_file.write(f"Folder: {folder_name}, Delay: {delay_str}\n")

        if delays:
            mean_delay = statistics.mean(delays)
            std_delay  = statistics.stdev(delays) if len(delays) > 1 else 0.0
            delay_file.write("\n")
            delay_file.write(f"Average Delay: {mean_delay:.2f} seconds\n")
            delay_file.write(f"Delay Std Dev: {std_delay:.2f} seconds\n")
        else:
            delay_file.write("\nNo valid delays found for statistics.\n")

    print(f"Delays and statistics saved to {delay_file_path}")
    print("Processing completed.")

if __name__ == "__main__":
    main()
