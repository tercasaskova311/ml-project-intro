# -------------------- IMPORTS --------------------
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- CONFIGURATION --------------------
K = 5  # Number of top similar images to retrieve
MODEL_NAME = "facebook/dinov2-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamically get paths based on where the file is
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_DIR = os.path.join(BASE_DIR, "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "data", "test", "gallery")
OUTPUT_PATH = os.path.join(BASE_DIR, "submissions", "sub_dino2.json")

# -------------------- LOAD MODEL --------------------
#We load the DINOv2 feature extractor (ViT-B/14) and set it to eval mode.

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# -------------------- TRANSFORM --------------------
#We resize and normalize the images the same way DINOv2 was trained.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])


# -------------------- DATASET --------------------
class ImagePathDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), os.path.basename(self.image_paths[idx])


# -------------------- FEATURE EXTRACTION --------------------
#his: Loads images in batches/Extracts embeddings using the model/Applies global average pooling/Returns the features and filenames

def extract_features(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_features, all_filenames = [], []

    with torch.no_grad():
        for images, filenames in tqdm(loader, desc="Extracting features"):
            images = images.to(DEVICE)
            feats = model(images).last_hidden_state.mean(dim=1)
            all_features.append(feats.cpu().numpy())
            all_filenames.extend(filenames)

    return np.vstack(all_features), all_filenames


# -------------------- MAIN SCRIPT --------------------
def main(k=K):
    print(f"[INFO] Starting DINOv2 Retrieval (top-{k})...")

    # Load datasets
    query_dataset = ImagePathDataset(QUERY_DIR, transform)
    gallery_dataset = ImagePathDataset(GALLERY_DIR, transform)

    # Extract features
    query_features, query_filenames = extract_features(query_dataset)
    gallery_features, gallery_filenames = extract_features(gallery_dataset)

    # Compute top-k similarity
    results = []
    for i, q_feat in enumerate(query_features):
        sims = cosine_similarity(q_feat.reshape(1, -1), gallery_features)[0]
        topk = np.argsort(sims)[::-1][:k]
        results.append({
            "filename": query_filenames[i],
            "samples": [gallery_filenames[j] for j in topk]
        })

    # Save result
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✅] Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
