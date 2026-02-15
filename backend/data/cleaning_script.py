import os
import shutil
import random

raw_images = "backend/data/images/images/train/"
raw_labels = "backend/data/labels/labels/train/"
output_base = "data"
image_ext = ".jpg"
random.seed(42)

# Mapping Class IDs to Names
class_map = {
    "0": "Deformation",
    "1": "Obstacle",
    "2": "Rupture",
    "3": "Disconnect",
    "4": "Misalignment",
    "5": "Deposition",
}

# Split Ratios: 70% Train, 15% Val, 15% Test
ratios = {"train": 0.70, "val": 0.15, "test": 0.15}


def process_and_split():
    # 1. Initialize temporary storage for valid file paths
    # We use a dictionary to group image paths by their class name
    valid_data = {name: [] for name in class_map.values()}
    deleted_count = 0

    print("Step 1: Cleaning and Validating Labels...")
    label_files = [f for f in os.listdir(raw_labels) if f.endswith(".txt")]

    for label_file in label_files:
        label_path = os.path.join(raw_labels, label_file)
        image_name = label_file.replace(".txt", image_ext)
        image_path = os.path.join(raw_images, image_name)

        if not os.path.exists(image_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        # Extract unique class IDs from the txt file
        class_ids = list(set([line.split()[0] for line in lines if line.strip()]))

        # LOGIC: Keep only if exactly 1 unique class exists in the file
        if len(class_ids) == 1:
            cid = class_ids[0]
            if cid in class_map:
                valid_data[class_map[cid]].append(
                    (image_path, label_path, image_name, label_file)
                )
            else:
                print(f"Warning: ID {cid} not in class_map. Skipping {label_file}")
        else:
            # Multi-class or empty files are "deleted" (ignored in the copy process)
            deleted_count += 1

    print(
        f"Cleaned! Kept: {sum(len(v) for v in valid_data.values())} | Filtered: {deleted_count}"
    )

    # 2. Create Directory Structure and Copy Files
    print("Step 2: Splitting and Organizing into Folders...")
    for split_name in ratios.keys():
        for class_name in class_map.values():
            os.makedirs(
                os.path.join(output_base, split_name, "Images", class_name),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(output_base, split_name, "Labels", class_name),
                exist_ok=True,
            )

    for class_name, files in valid_data.items():
        random.shuffle(files)

        total = len(files)
        train_end = int(total * ratios["train"])
        val_end = train_end + int(total * ratios["val"])

        split_files = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }

        for split_name, file_list in split_files.items():
            for img_p, lbl_p, img_n, lbl_n in file_list:
                # Copy Image
                shutil.copy2(
                    img_p,
                    os.path.join(output_base, split_name, "Images", class_name, img_n),
                )
                # Copy Label (Backup)
                shutil.copy2(
                    lbl_p,
                    os.path.join(output_base, split_name, "Labels", class_name, lbl_n),
                )

    print(f"\nSuccess! Your dataset is ready in: {os.path.abspath(output_base)}")


if __name__ == "__main__":
    process_and_split()
