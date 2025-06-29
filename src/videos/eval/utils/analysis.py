import matplotlib.pyplot as plt
import json
import os

def calculate_detection_metrics(detection_data):
    before_ignition_total = detection_data["before_ignition_not_detected"] + detection_data["before_ignition_detected"]
    after_ignition_total = detection_data["after_ignition_not_detected"] + detection_data["after_ignition_detected"]

    precision = (detection_data["after_ignition_detected"] / (detection_data["after_ignition_detected"] + detection_data["before_ignition_detected"])) if (detection_data["after_ignition_detected"] + detection_data["before_ignition_detected"]) > 0 else None

    recall = detection_data["after_ignition_detected"] / after_ignition_total if after_ignition_total > 0 else None

    f1_score = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    metrics = {
        "TN_before_ignition": detection_data["before_ignition_not_detected"],
        
        "FP_before_ignition": detection_data["before_ignition_detected"],
        
        "FN_after_ignition": detection_data["after_ignition_not_detected"],
        
        "TP_after_ignition": detection_data["after_ignition_detected"],

        "detection_delay_seconds": detection_data["detection_delay"] if detection_data["detection_delay"] is not None else -1,

        "recall": recall,

        "precision": precision,

        "f1_score": f1_score
    }

    return metrics


def save_metrics_as_txt(metrics, output_folder, filename='results.txt'):
    with open(os.path.join(output_folder, filename), 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_metrics(metrics, output_folder, filename='results.png'):
    labels = list(metrics.keys())
    values = [0 if v is None else v for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, values, color='skyblue')
    ax.set_xlabel('Count / Seconds')
    ax.set_title('Detection Metrics')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()
