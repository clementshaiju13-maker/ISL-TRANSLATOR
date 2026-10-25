import os
import shutil

# Put the path where you unzipped the archive here:
SOURCE_UNZIPPED_DIR = r"D:\isl\archive"  # Change to your unzipped folder path
TARGET_DIR = r"D:\isl\dataset_raw"
IMAGES_PER_LETTER = 100
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)  # Clear the old folder cleanly
os.makedirs(TARGET_DIR, exist_ok=True)

# Scan through all subfolders in the unzipped folder
subfolders = sorted([f for f in os.listdir(SOURCE_UNZIPPED_DIR) if os.path.isdir(os.path.join(SOURCE_UNZIPPED_DIR, f))])

for idx, folder_name in enumerate(subfolders):
    src_folder = os.path.join(SOURCE_UNZIPPED_DIR, folder_name)
    dst_folder = os.path.join(TARGET_DIR, f"dataset_raw-{idx + 1}")
    os.makedirs(dst_folder, exist_ok=True)

    all_images = [f for f in os.listdir(src_folder) if f.lower().endswith(VALID_EXTS)]
    selected_images = all_images[:IMAGES_PER_LETTER]

    for img_name in selected_images:
        shutil.copy(os.path.join(src_folder, img_name), os.path.join(dst_folder, img_name))

    print(f"Copied {len(selected_images)} images from '{folder_name}' -> 'dataset_raw-{idx + 1}'")

print("\nDone! Clean 100-images-per-letter dataset created.")