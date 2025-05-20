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

    zip_files = [f for f in os.listdir(dest_folder) if f.endswith(".zip")]
    if not zip_files:
        raise FileNotFoundError("No .zip file found after download.")
    
    zip_path = os.path.join(dest_folder, zip_files[0])
    print("Extracting contents...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)  # Extracts directly to `data/`
    os.remove(zip_path)



def organize_dataset_by_percentage(base_path="data", train_pct=0.75, query_pct=0.005, gallery_pct=0.245):
    """
    Splits existing images inside 'data/training/<class>/' into:
    - 'data/training/<class>/'
    - 'data/test/query/<class>/'
    - 'data/test/gallery/<class>/'
    """

    base_path = Path(base_path)
    training_dir = base_path / "training"
    test_query_dir = base_path / "test" / "query"
    test_gallery_dir = base_path / "test" / "gallery"

    # Clean and recreate test folders
    for path in [test_query_dir, test_gallery_dir]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    # Go through each class inside data/training/
    class_dirs = [d for d in training_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.*"))
        if len(images) == 0:
            continue

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_pct)
        n_query = int(n_total * query_pct)
        n_gallery = n_total - n_train - n_query

        # Split images
        train_imgs = images[:n_train]
        query_imgs = images[n_train:n_train + n_query]
        gallery_imgs = images[n_train + n_query:]

        # Clear and recreate current training class folder
        shutil.rmtree(class_dir)
        class_dir.mkdir(parents=True)

        # Copy training images back into training/
        for img in train_imgs:
            shutil.copy(img, class_dir / img.name)

        # Copy query images to test/query/<class>/
        query_dst = test_query_dir / class_dir.name
        query_dst.mkdir(parents=True, exist_ok=True)
        for img in query_imgs:
            shutil.copy(img, query_dst / img.name)

        # Copy gallery images to test/gallery/<class>/
        gallery_dst = test_gallery_dir / class_dir.name
        gallery_dst.mkdir(parents=True, exist_ok=True)
        for img in gallery_imgs:
            shutil.copy(img, gallery_dst / img.name)

        print(f"{class_dir.name}: Total={n_total}, Train={len(train_imgs)}, Query={len(query_imgs)}, Gallery={len(gallery_imgs)}")

    print("✅ Dataset successfully reorganized by percentage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and organize Kaggle dataset")
    # parser.add_argument("--dataset", required=True, help="Kaggle dataset name (e.g. 'prasunroy/natural-images')")
    parser.add_argument("--out_dir", default="data", help="Output folder (default: data)")
    args = parser.parse_args()

    # download_and_extract(args.dataset, args.out_dir)
    organize_dataset_by_percentage(args.out_dir)
