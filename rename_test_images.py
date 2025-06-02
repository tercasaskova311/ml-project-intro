import os
import shutil
from glob import glob
from collections import defaultdict

# Config paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(BASE_DIR, "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data", "test", "gallery")
QUERY_OUT_DIR = os.path.join(BASE_DIR, "data", "test", "query_flat")
GALLERY_OUT_DIR = os.path.join(BASE_DIR, "data", "test", "gallery_flat")

# Create output directories
os.makedirs(QUERY_OUT_DIR, exist_ok=True)
os.makedirs(GALLERY_OUT_DIR, exist_ok=True)

def flatten_and_rename(src_root, dst_root):
    class_counters = defaultdict(int)

    for class_folder in os.listdir(src_root):
        class_path = os.path.join(src_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        for img_path in glob(os.path.join(class_path, "*")):
            ext = os.path.splitext(img_path)[1]
            class_counters[class_folder] += 1
            new_name = f"{class_folder}_{class_counters[class_folder]}{ext}"
            dst_path = os.path.join(dst_root, new_name)
            shutil.copy(img_path, dst_path)

    print(f"✅ Copiate e rinominate {sum(class_counters.values())} immagini da {src_root} a {dst_root}")

# Run for both query and gallery
flatten_and_rename(QUERY_DIR, QUERY_OUT_DIR)
flatten_and_rename(GALLERY_DIR, GALLERY_OUT_DIR)
