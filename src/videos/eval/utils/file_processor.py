import os
import cv2
from datetime import datetime, timedelta, time
from tqdm import tqdm
from .ignition_time import get_ignition_time
from .analysis import calculate_detection_metrics, save_metrics_as_txt, plot_metrics
from models.model_loader import load_model
import uuid 

DATE_TIME_LEN = 19

def expand_bounding_box(x1, y1, x2, y2, img_width, img_height, percentage=0.1):
    area_to_expand = img_width * img_height * percentage

    width_expand = (area_to_expand / ((y2 - y1) + 1)) ** 0.5
    height_expand = (area_to_expand / ((x2 - x1) + 1)) ** 0.5

    x1_new = max(0, x1 - width_expand / 2)
    y1_new = max(0, y1 - height_expand / 2)
    x2_new = min(img_width, x2 + width_expand / 2)
    y2_new = min(img_height, y2 + height_expand / 2)

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

def is_bounding_box_large_enough(x1, y1, x2, y2, min_width=20, min_height=20, max_width=200, max_height=200):
    width = x2 - x1
    height = y2 - y1

    return min_width <= width <= max_width and min_height <= height <= max_height

def crop_previous_frames(image_paths, bounding_box, temp_dir):
    cropped_images_paths = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue
        x1, y1, x2, y2 = bounding_box
        cropped_image = image[y1:y2, x1:x2]  

        temp_image_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
        cv2.imwrite(temp_image_path, cropped_image) 
        cropped_images_paths.append(temp_image_path)
    return cropped_images_paths

def process_files_in_folder(folder_path, output_folder, yolo_model, lstm_resnet_model, ignition_times, confidence_threshold, frames_back):
    folder_name = os.path.basename(folder_path)
    ignition_time = get_ignition_time(folder_name, ignition_times)

    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.jpg')],
        key=lambda filename: datetime.strptime(filename[-(DATE_TIME_LEN+4):-4], '%Y_%m_%dT%H_%M_%S')
    )

    print(f"Processing folder: {folder_path}")
    print(f"Number of files: {len(file_list)}")

    if not file_list:
        return None

    first_filename = file_list[0]
    date_time_str = first_filename[-(DATE_TIME_LEN+4):-4]

    detection_data = {
        "folder_name": folder_name,
        "before_ignition_not_detected": 0,
        "before_ignition_detected": 0,
        "after_ignition_not_detected": 0,
        "after_ignition_detected": 0,
        "detection_delay": None,
        "total_frames": len(file_list),
        "bounding_boxes": [],
        "frame_predictions": []
    }

    detected_after_ignition = False

    detections_folder = os.path.join(output_folder, 'detections')
    results_folder = os.path.join(output_folder, 'results')
    zoom_folder = os.path.join(output_folder, 'data/buffer')
    temp_crop_folder = os.path.join(output_folder, 'temp_crops')
    os.makedirs(detections_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(zoom_folder, exist_ok=True)
    os.makedirs(temp_crop_folder, exist_ok=True)

    def clear_buffer_folder(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    for frame_idx, filename in enumerate(tqdm(file_list, desc=f"Processing images in {folder_name}")):
        current_time = datetime.strptime(filename[-(DATE_TIME_LEN+4):-4], '%Y_%m_%dT%H_%M_%S')
        file_path = os.path.join(folder_path, filename)
        print(f"Processing image: {file_path}")
        image = cv2.imread(file_path)
        print(f"Image shape: {image.shape}")
        img_height, img_width = image.shape[:2]
        results = yolo_model(file_path, verbose=False, conf=confidence_threshold)
        num_boxes = len(results[0].boxes)
        print(f"Número de bounding boxes detectados: {num_boxes}")

        detected = False 

        if num_boxes > 0 and lstm_resnet_model:
            boxes = results[0].boxes.xyxy
            confidences = results[0].boxes.conf

            for i in range(num_boxes):
                x1, y1, x2, y2 = map(int, boxes[i])

                if not is_bounding_box_large_enough(x1, y1, x2, y2):
                    continue

                expanded_box = expand_bounding_box(x1, y1, x2, y2, img_width, img_height)

                previous_images = file_list[max(0, frame_idx - frames_back):frame_idx]
                previous_image_paths = [os.path.join(folder_path, f) for f in previous_images]
                previous_image_paths.append(file_path)

                cropped_images_paths = crop_previous_frames(previous_image_paths, expanded_box, temp_crop_folder)
                if len(cropped_images_paths) == frames_back+1:
                    global_prediction = lstm_resnet_model.infer_5_frames(cropped_images_paths)
                    print(f"Global prediction for frame {frame_idx}: {global_prediction}")
                    if global_prediction > 0.5:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        detected = True 
                        output_image_path = os.path.join(detections_folder, filename)
                        cv2.imwrite(output_image_path, image)
                        print(f"Frame index: {frame_idx}, Image name: {filename}, Detected: {detected}")
        
        elif num_boxes > 0 and lstm_resnet_model is None:
            boxes = results[0].boxes.xyxy
            detection_data["bounding_boxes"].append([int(x) for box in boxes for x in box])

            detected = True

            for i in range(num_boxes):
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output_image_path = os.path.join(detections_folder, filename)
            cv2.imwrite(output_image_path, image)

        
        print(f"Frame index: {frame_idx}, Image name: {filename}, Detected: {detected}")

        if ignition_time is None:
            ignition_time = datetime.max.time()
            ignition_time = datetime.combine(datetime.max.date(), ignition_time)
            print("NO WILDFIRE")

        print(f"Current time: {current_time}, Ignition time: {ignition_time}")
        
        if current_time < ignition_time:
            if detected:
                detection_data["before_ignition_detected"] += 1
                detection_data["frame_predictions"].append(-1)
            else:
                detection_data["before_ignition_not_detected"] += 1
                detection_data["frame_predictions"].append(1)

        elif current_time == ignition_time:
            print("*-"*30)
            if 0 not in detection_data["frame_predictions"]:
                detection_data["frame_predictions"].append(0)
                print("Ignition frame detected")
            if detected:
                detection_data["after_ignition_detected"] += 1
                detection_data["frame_predictions"].append(1)
                print("Ignition frame detected correctly")
            else:
                detection_data["after_ignition_not_detected"] += 1
                detection_data["frame_predictions"].append(-1)

        else:
            if detected:
                detection_data["after_ignition_detected"] += 1
                detection_data["frame_predictions"].append(1)
            else:
                detection_data["after_ignition_not_detected"] += 1
                detection_data["frame_predictions"].append(-1)
        if current_time == ignition_time and not detected_after_ignition:
            detected_after_ignition = True

        if detected_after_ignition and detected and detection_data["detection_delay"] is None:
            print(f"Detection time: {current_time}")
            print(f"Ignition time: {ignition_time}")
            d_delay = current_time - ignition_time
            print(f"Detection delay: {d_delay}")
            d_delay = timedelta(hours=d_delay.seconds // 3600, minutes=(d_delay.seconds // 60) % 60)
            print(f"Detection delay: {d_delay}")
            d_delay = d_delay.total_seconds()
            print(f"Detection delay: {d_delay}")
            d_delay = int(d_delay) // 60
            print(f"Detection delay: {d_delay}")
            detection_data["detection_delay"] = d_delay

    clear_buffer_folder(temp_crop_folder)

    metrics = calculate_detection_metrics(detection_data)
    save_metrics_as_txt(metrics, results_folder)
    plot_metrics(metrics, results_folder)

    return detection_data
