import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_training_to_train_test(base_dir, train_pct=0.75, query_pct=0.005, gallery_pct=0.245):
    base_dir = Path(base_dir)
    training_dir = base_dir / "training"
    test_dir = base_dir / "test"
    test_query_dir = test_dir / "query"
    test_gallery_dir = test_dir / "gallery"

    # Delete existing folders
    for folder in [training_dir, test_dir]:
        if folder.exists():
            shutil.rmtree(folder)

    # Recreate folders
    training_dir.mkdir(parents=True, exist_ok=True)
    test_query_dir.mkdir(parents=True, exist_ok=True)
    test_gallery_dir.mkdir(parents=True, exist_ok=True)

    # Gather all images from original training data
    original_training_dir = base_dir / "training_original"
    if not original_training_dir.exists():
        raise FileNotFoundError(f"Expected original training folder at {original_training_dir}.")

    all_images = []
    for class_dir in original_training_dir.iterdir():
        if class_dir.is_dir():
            imgs = list(class_dir.glob("*.*"))
            all_images.extend([(img, class_dir.name) for img in imgs])

    if not all_images:
        raise ValueError("No images found in original training folder.")

    total = len(all_images)
    print(f"Total images found: {total}")

    random.shuffle(all_images)

    n_train = int(total * train_pct)
    n_query = int(total * query_pct)
    n_gallery = int(total * gallery_pct)

    # Sanity check
    if n_train + n_query + n_gallery > total:
        raise ValueError("Sum of splits exceeds total images.")

    train_imgs = all_images[:n_train]
    query_imgs = all_images[n_train:n_train + n_query]
    gallery_imgs = all_images[n_train + n_query:n_train + n_query + n_gallery]

    # Copy train images preserving class folders
    print(f"Copying {len(train_imgs)} train images...")
    for img_path, cls in tqdm(train_imgs):
        dest_dir = training_dir / cls
        dest_dir.mkdir(exist_ok=True)
        shutil.copy(img_path, dest_dir / img_path.name)

    # Copy query images flat
    print(f"Copying {len(query_imgs)} query images...")
    for img_path, _ in tqdm(query_imgs):
        shutil.copy(img_path, test_query_dir / img_path.name)

    # Copy gallery images flat
    print(f"Copying {len(gallery_imgs)} gallery images...")
    for img_path, _ in tqdm(gallery_imgs):
        shutil.copy(img_path, test_gallery_dir / img_path.name)

    print("Done splitting dataset.")

if __name__ == "__main__":
    # Assumes you have the original full training dataset in "data/training_original"
    split_training_to_train_test("data")
