import os
import glob
import random
import shutil
import re
from datetime import datetime

# Configuration
random.seed(16)

# Base paths (replace these with your actual dataset locations)
DATASET_PATHS = {
    "datav3": "/path/to/dataset/video_ds",
    "new_ds_fp": "/path/to/dataset/video_ds_fp",
}
OUTPUT_PATH = "output_dataset"

# Parameters
GROUP_MIN_SIZE = 4          # Minimum number of files in a group
TIME_THRESHOLD = 2 * 60     # Time threshold (in seconds) for grouping files
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def find_label_files(base_path):
    """Find all label files under a base path."""
    label_files = glob.glob(os.path.join(base_path, "**", "labels", "*.txt"), recursive=True)
    label_files.sort()
    print(f"Found {len(label_files)} label files in {base_path}")
    return label_files


def group_files_by_time(label_files):
    """Group files by their timestamp proximity."""
    groups = {}
    current_group = -1
    last_time = datetime.now()

    for file in label_files:
        match = re.search(r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})", file)
        if match:
            timestamp = datetime.strptime(match.group(), "%Y_%m_%dT%H_%M_%S")
            if abs((timestamp - last_time).total_seconds()) > TIME_THRESHOLD:
                current_group += 1
            last_time = timestamp

            groups.setdefault(current_group, []).append(file)

    print(f"Created {len(groups)} time-based groups")
    return groups


def filter_groups(groups):
    """Remove groups that don't meet the minimum file count or have empty label files."""
    valid = {}
    for gid, files in groups.items():
        if len(files) >= GROUP_MIN_SIZE:
            non_empty = [f for f in files if len(open(f).readlines()) > 0]
            if len(non_empty) >= GROUP_MIN_SIZE:
                valid[gid] = non_empty

    print(f"{len(valid)} groups remain after filtering")
    return valid


def split_and_save_groups(groups, output_path, train_ratio, val_ratio, test_ratio):
    """Split groups into train, val, and test sets and copy files accordingly."""
    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    # Calculate split indices
    n = len(group_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_ids = group_ids[:train_end]
    val_ids = group_ids[train_end:val_end]
    test_ids = group_ids[val_end:]

    print(f"Splitting into {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test groups")

    used_files = set()

    def save_subset(ids, subset_name):
        for gid in ids:
            group_files = [f for f in groups[gid] if f not in used_files]
            if len(group_files) >= GROUP_MIN_SIZE:
                selection = group_files[:GROUP_MIN_SIZE]
                for label_file in selection:
                    _copy_pair(label_file, subset_name, gid)
                    used_files.add(label_file)

    def _copy_pair(label_file, subset_name, group_id):
        # Copy label
        label_dest = os.path.join(output_path, "labels", subset_name, "0", str(group_id))
        os.makedirs(label_dest, exist_ok=True)
        shutil.copy(label_file, os.path.join(label_dest, os.path.basename(label_file)))

        # Copy corresponding image
        img_file = label_file.replace("labels/", "").replace(".txt", ".jpg")
        img_dest = os.path.join(output_path, "images", subset_name, "0", str(group_id))
        os.makedirs(img_dest, exist_ok=True)
        shutil.copy(img_file, os.path.join(img_dest, os.path.basename(img_file)))

    # Perform copies for each split
    save_subset(train_ids, "train")
    save_subset(val_ids, "val")
    save_subset(test_ids, "test")


def count_files_in_sets(output_path):
    """Count label and image files in each subset folder."""
    subsets = ["train", "val", "test"]
    summary = {}

    for subset in subsets:
        lbl_path = os.path.join(output_path, "labels", subset)
        img_path = os.path.join(output_path, "images", subset)

        n_labels = len(glob.glob(os.path.join(lbl_path, "**", "*.txt"), recursive=True))
        n_images = len(glob.glob(os.path.join(img_path, "**", "*.jpg"), recursive=True))

        summary[subset] = {"labels": n_labels, "images": n_images}

    print("\nFile counts in output_dataset:")
    for subset, counts in summary.items():
        print(f"{subset.title()}: {counts['labels']} labels, {counts['images']} images")

    return summary


if __name__ == "__main__":
    # Main execution: process each dataset path
    for name, path in DATASET_PATHS.items():
        print(f"\nProcessing dataset: {name}")
        files = find_label_files(path)
        grouped = group_files_by_time(files)
        valid = filter_groups(grouped)
        split_and_save_groups(valid, OUTPUT_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Final summary
    count_files_in_sets(OUTPUT_PATH)
