import os
import json
from PIL import Image
import matplotlib.pyplot as plt

# Config
submission_name = input("Enter the submission filename (e.g., submission_resnet.json): ").strip()
if not submission_name.endswith('.json'):
    submission_name += '.json'  

submission_path = os.path.join("submissions", submission_name)  # path to the JSON file
query_dir = "data/test/query"                         # path to query images
gallery_dir = "data/test/gallery"                     # path to gallery images
max_queries_to_show = 20                              # how many query images to visualize

# Load JSON
with open(submission_path, 'r') as f:
    submission = json.load(f)

# Visualize
for i, (query_file, retrieved_files) in enumerate(submission.items()):
    if i >= max_queries_to_show:
        break

    query_img_path = os.path.join(query_dir, query_file)
    if not os.path.exists(query_img_path):
        print(f"[Warning] Query image not found: {query_img_path}")
        continue

    query_img = Image.open(query_img_path).convert("RGB")
    retrieved_imgs = []

    for fname in retrieved_files:
        found = False
        for root, _, files in os.walk(gallery_dir):
            if fname in files:
                retrieved_imgs.append(Image.open(os.path.join(root, fname)).convert("RGB"))
                found = True
                break
        if not found:
            print(f"[Warning] Could not find {fname} in gallery!")

    if not retrieved_imgs:
        print(f"[Info] No retrieved images found for {query_file}, skipping plot.")
        continue

    # Plot
    n = len(retrieved_imgs)
    plt.figure(figsize=(2 + n, 2))
    plt.subplot(1, n + 1, 1)
    plt.imshow(query_img)
    plt.axis('off')
    plt.title("Query")

    for j, img in enumerate(retrieved_imgs):
        plt.subplot(1, n + 1, j + 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Top {j+1}")

    plt.tight_layout()
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f"visualizations/query_{i:03}.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved visualization for query {query_file}")


    #plt.savefig('plot.png', bbox_inches='tight', dpi=300)  # Save the plot if needed