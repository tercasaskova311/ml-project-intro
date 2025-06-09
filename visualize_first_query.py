import json
import os
import matplotlib.pyplot as plt
from PIL import Image

# -------- PATHS --------
query_dir = "./data/test/query"
gallery_dir = "./data/test/gallery"
submission_path = "./submissions/sub_clip_data.json"
output_path = "retrieval_result.png"

# -------- LOAD SUBMISSION --------
with open(submission_path, "r") as f:
    submission = json.load(f)

# -------- GET FIRST QUERY --------
first_query, retrieved_images = next(iter(submission.items()))

# -------- LOAD QUERY IMAGE --------
query_img = Image.open(os.path.join(query_dir, first_query)).convert("RGB")

# -------- LOAD TOP-10 GALLERY IMAGES --------
retrieved_imgs = [Image.open(os.path.join(gallery_dir, img)).convert("RGB") for img in retrieved_images]

# -------- PLOT RESULTS --------
plt.figure(figsize=(15, 3))
plt.subplot(1, 11, 1)
plt.imshow(query_img)
plt.title("Query")
plt.axis("off")

for i, img in enumerate(retrieved_imgs[:10]):
    plt.subplot(1, 11, i + 2)
    plt.imshow(img)
    plt.title(f"Top {i+1}")
    plt.axis("off")

# -------- SAVE OUTPUT --------
plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved to {output_path}")
