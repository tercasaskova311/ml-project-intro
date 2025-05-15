#on terminal: python src/visualize_submission.py

import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Config
submission_path = "submissions/submission_dino.json"  # path to the JSON file
query_dir = "data/test/query"                         # path to query images
gallery_dir = "data/training"                         # path to training images
max_queries_to_show = 5                               # how many query images to visualize

# Load JSON
with open(submission_path, 'r') as f:
    submission = json.load(f)

# Visualize
for i, entry in enumerate(submission[:max_queries_to_show]):
    query_file = entry['filename']
    retrieved_files = entry['samples']

    # Load query image
    query_img_path = os.path.join(query_dir, query_file)
    query_img = Image.open(query_img_path).convert("RGB")

    # Load retrieved images
    retrieved_imgs = []
    for fname in retrieved_files:
        # Since gallery may be nested in folders, search all subfolders
        found = False
        for root, _, files in os.walk(gallery_dir):
            if fname in files:
                retrieved_imgs.append(Image.open(os.path.join(root, fname)).convert("RGB"))
                found = True
                break
        if not found:
            print(f"[Warning] Could not find {fname} in training data!")

    # Plot
    n = len(retrieved_imgs)
    plt.figure(figsize=(2 + n, 2))

    # Query image
    plt.subplot(1, n + 1, 1)
    plt.imshow(query_img)
    plt.axis('off')
    plt.title("Query")

    # Retrieved images
    for j, img in enumerate(retrieved_imgs):
        plt.subplot(1, n + 1, j + 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Top {j+1}")

    plt.tight_layout()
    plt.show()
