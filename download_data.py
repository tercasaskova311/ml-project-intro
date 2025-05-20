import os
import argparse
import zipfile
import random
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from tqdm import tqdm

def download_and_extract(dataset_name, dest_folder):
    api = KaggleApi()
    api.authenticate()

    os.makedirs(dest_folder, exist_ok=True)
    print(f"Downloading: {dataset_name}")
    api.dataset_download_files(dataset_name, path=dest_folder, quiet=False)

    # Locate the downloaded zip file
    zip_files = [f for f in os.listdir(dest_folder) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError("No .zip file found after download.")
    
    zip_path = os.path.join(dest_folder, zip_files[0])
    print("Extracting contents...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)

def organize_dataset_by_percentage(dest_folder, train_pct=0.75, query_pct=0.005, gallery_pct=0.245):
    print("Organizing dataset into training/ and test/ folders (percentage-based)...")
    base_data_dir = Path(dest_folder)

    # Ask user before deleting existing training/test folders
    for folder in ["training", "test"]:
        folder_path = base_data_dir / folder
        if folder_path.exists():
            response = input(f"Folder '{folder_path}' already exists and will be deleted. Continue? (y/n): ").strip().lower()
            if response == 'y':
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")
            else:
                print(f"Aborted. Folder '{folder_path}' was not removed.")
                exit(1)

    # Create final directory structure
    training_dir = base_data_dir / "training"
    test_query_dir = base_data_dir / "test" / "query"
    test_gallery_dir = base_data_dir / "test" / "gallery"

    for path in [training_dir, test_query_dir, test_gallery_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Gather all images from class folders
    extracted_dirs = [d for d in base_data_dir.iterdir() if d.is_dir() and d.name not in ["training", "test"]]
    all_images = []
    for class_dir in extracted_dirs:
        all_images.extend(list(class_dir.glob("*.*")))

    total = len(all_images)
    if total == 0:
        raise ValueError("No images found!")

    # Shuffle and split the dataset
    random.shuffle(all_images)
    n_train = int(total * train_pct)
    n_query = int(total * query_pct)
    n_gallery = int(total * gallery_pct)

    if n_train + n_query + n_gallery > total:
        raise ValueError("The sum of splits exceeds the total number of images.")

    train_imgs = all_images[:n_train]
    query_imgs = all_images[n_train:n_train + n_query]
    gallery_imgs = all_images[n_train + n_query:n_train + n_query + n_gallery]

    print(f"Total images: {total}")
    print(f"Training: {len(train_imgs)}, Query: {len(query_imgs)}, Gallery: {len(gallery_imgs)}")

    # Copy to training folder
    for img in tqdm(train_imgs, desc="Copying training"):
        class_name = img.parent.name
        dst = training_dir / class_name
        dst.mkdir(exist_ok=True)
        shutil.copy(img, dst / img.name)

    # Copy to query folder
    for img in tqdm(query_imgs, desc="Copying query"):
        class_name = img.parent.name
        dst = test_query_dir / class_name
        dst.mkdir(exist_ok=True)
        shutil.copy(img, dst / img.name)

    # Copy to gallery folder
    for img in tqdm(gallery_imgs, desc="Copying gallery"):
        class_name = img.parent.name
        dst = test_gallery_dir / class_name
        dst.mkdir(exist_ok=True)
        shutil.copy(img, dst / img.name)

    # Clean up the extracted raw folders
    for d in extracted_dirs:
        shutil.rmtree(d)

    print("Dataset organized (75% train, 0.5% query, 24.5% gallery)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize Kaggle dataset")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset name (e.g. 'alxmamaev/flowers-recognition')")
    parser.add_argument("--out_dir", default="data", help="Output folder (default: data)")
    args = parser.parse_args()

    download_and_extract(args.dataset, args.out_dir)
    organize_dataset_by_percentage(args.out_dir)
