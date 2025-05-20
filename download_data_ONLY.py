import os
from kaggle.api.kaggle_api_extended import KaggleApi

# ----------------------------------------
# 1. Setup: Check and create the destination folder
# ----------------------------------------
dataset_slug = "iamsouravbanerjee/animal-image-dataset-90-different-animals"  # Example dataset
destination_path = "data"

os.makedirs(destination_path, exist_ok=True)

# ----------------------------------------
# 2. Download using Kaggle API
# ----------------------------------------
api = KaggleApi()
api.authenticate()

print(f"Downloading {dataset_slug}...")
api.dataset_download_files(dataset=dataset_slug, path=destination_path, unzip=True)
print("Download complete.")

# later manually set up a folder with all images named training_original and run the splitting.py script!!!